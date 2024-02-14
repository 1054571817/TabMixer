import time
import glob
import os
import warnings
from natsort import natsorted
from utils import load_yaml

global_args = load_yaml("params_global.yml")
if global_args.project_name == "ph_4ch":
    params_path = 'params_ph_4ch.yml'
elif global_args.project_name == "ph_sa":
    params_path = 'params_ph_sa.yml'
args = load_yaml(params_path)

args.data.cache_dir = os.path.join(args.data.cache_dir,
                                   f"cuda{args.gpu.cuda_device_id}__{args.data.augmentation.resize_size[0]}_{args.data.augmentation.resize_size[1]}_{args.data.augmentation.resize_size[2]}")

if args.data.clear_cache:
    print("Clearning cache...")
    train_cache = glob.glob(os.path.join(args.data.cache_dir, 'train/*.pt'))
    val_cache = glob.glob(os.path.join(args.data.cache_dir, 'val/*.pt'))
    test_cache = glob.glob(os.path.join(args.data.cache_dir, 'test/*.pt'))
    if len(train_cache) != 0:
        for file in train_cache:
            os.remove(file)
    if len(val_cache) != 0:
        for file in val_cache:
            os.remove(file)
    if len(test_cache) != 0:
        for file in test_cache:
            os.remove(file)
    print(
        f"Cleared cache in dir: {args.data.cache_dir}, train: {len(train_cache)} files, val: {len(val_cache)} files, test: {len(test_cache)} files.")

# TORCH modules
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import RandomSampler
from torch.nn import MSELoss
# MONAI modules
from monai.metrics import MAEMetric, MSEMetric, RMSEMetric
from monai.metrics import CumulativeAverage
from monai.optimizers import WarmupCosineSchedule
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import set_track_meta, ThreadDataLoader
from monai.data.dataset import PersistentDataset

# models implementations
from models.daft_paper_models import DAFT_MODEL, FILM_MODEL, INTERACTIVE_MODEL, HeterogeneousResNet, ConcatHNN1FC, \
    ConcatHNN2FC, ConcatHNNMCM, TABATTENTION_MODEL, TABMIXER_MODEL
from models.resnet import r3d_18
from models.inceptioni3d import InceptionI3d
from models.swintransformer import SwinTransformer3D
from models.mlp3d import MLP3D_T
from metrics import compute_metrics

if global_args.comet.print_config:
    print_config()

# external modules
import pandas as pd
from cuda import setup_cuda

from data_augmentation_ph import Transforms
from tabular_data import TabularDataLoader, get_tabular_config

# use amp to accelerate training
scaler = None
if args.optimizer.scaler.use_scaler:
    scaler = torch.cuda.amp.GradScaler()
TORCH_DTYPES = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64
}

autocast_d_type = TORCH_DTYPES[args.optimizer.scaler.autocast_dtype]
if autocast_d_type == torch.bfloat16:
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


# CUDA
if args.gpu.device == 'cuda':
    setup_cuda(args.gpu.gpu_frac, num_threads=args.gpu.num_threads, device=args.gpu.device,
               visible_devices=args.gpu.visible_devices,
               use_cuda_with_id=args.gpu.cuda_device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=int(args.gpu.cuda_device_id))
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(int(args.gpu.cuda_device_id))

# TRANSFORMS
trans = Transforms(args, device)
set_track_meta(True)

# MONAI backed dataset
data_root_dir = args.data.videos_path

tdl = TabularDataLoader(args.data.tabular_data_path)

if not os.path.exists(args.data.cache_dir):
    os.makedirs(os.path.join(args.data.cache_dir, 'train'))
    os.makedirs(os.path.join(args.data.cache_dir, 'val'))
    os.makedirs(os.path.join(args.data.cache_dir, 'test'))

train_datasets = [PersistentDataset()] # TODO prepare dataset
val_datasets = [PersistentDataset()] # TODO prepare dataset
test_dataset = PersistentDataset() # TODO prepare dataset

if args.optimizer.loss == "MSE":
    criterion = MSELoss()
else:
    raise NotImplementedError(f"No implementation for {args.loss_name}")


## TRAINING_STEP
def training_step(batch_idx, train_data, args):
    with torch.cuda.amp.autocast(enabled=args.optimizer.scaler.use_scaler, dtype=autocast_d_type):
        if args.model.tabular_data:
            output = model(train_data["image"], train_data["tabular"].to(device))
        else:
            output = model(train_data["image"])
        loss = criterion(output.float(), train_data["y"].float().to(device))

    if args.optimizer.scaler.use_scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pred = torch.reshape(output.float(), (-1, 1))
    y_pred = torch.reshape(train_data["y"].float(), (-1, 1)).to(device)

    train_loss_cum.append(loss.item(), count=train_data["image"].shape[0])

    for func in metrics:
        func(y_pred=pred, y=y_pred)

    if (batch_idx + 1) == len(train_loader):
        metric_results = [func.aggregate().mean().item() for func in metrics]

        # log running average for metrics
        train_mae_cum.append(metric_results[0])
        train_rmse_cum.append(metric_results[1])
        train_mse_cum.append(metric_results[2])

        print(f" Train metrics:\n"
              f"  * MAE: {metric_results[0]:.3f}, RMSE: {metric_results[1]:.3f}.")

    epoch_time = time.time() - start_time_epoch

    if (batch_idx + 1) % args.intervals.log_batch == 0:
        print(" ", end="")
        print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
              f" Loss: {loss.item():.4f}."
              f" Time: {epoch_time:.2f}s")


### VALIDATION STEP ###
def validation_step(batch_idx, val_data, args):
    with torch.cuda.amp.autocast(enabled=args.optimizer.scaler.use_scaler, dtype=autocast_d_type):
        if args.model.tabular_data:
            val_output = model(val_data["image"], val_data["tabular"].to(device))
        else:
            val_output = model(val_data["image"])

        val_preds = torch.reshape(val_output, (-1, 1))
        val_labels = torch.reshape(val_data["y"].to(device), (-1, 1))

        for func in metrics:
            func(y_pred=val_preds, y=val_labels)

        if (batch_idx + 1) == len(val_loader):
            metric_results = [func.aggregate().mean().item() for func in metrics]

            # log running average for metrics
            val_mae_cum.append(metric_results[0])
            val_rmse_cum.append(metric_results[1])
            val_mse_cum.append(metric_results[2])

            print(f" Validation metrics:\n"
                  f"  * Reg.: mae: {metric_results[0]:.3f}, rmse: {metric_results[1]:.3f}.")


### TEST STEP ###
def test_step(batch_idx, test_data, args):
    with torch.cuda.amp.autocast(enabled=args.optimizer.scaler.use_scaler, dtype=autocast_d_type):
        if args.model.tabular_data:
            test_output = model(test_data["image"], test_data["tabular"].to(device))
        else:
            test_output = model(test_data["image"])

        test_preds = torch.reshape(test_output, (-1, 1))
        test_labels = torch.reshape(test_data["y"].to(device), (-1, 1))
        for func in metrics:
            func(y_pred=test_preds, y=test_labels)

        if (batch_idx + 1) == len(test_loader):
            metric_results = [func.aggregate().mean().item() for func in metrics]

            # log running average for metrics
            test_mae_cum.append(metric_results[0])
            test_rmse_cum.append(metric_results[1])
            test_mse_cum.append(metric_results[2])

            print(f" Test metrics:\n"
                  f"  * Reg.: mae: {metric_results[0]:.3f}, rmse: {metric_results[1]:.3f}.")
        return test_preds.flatten().detach().cpu().numpy(), test_labels.flatten().detach().cpu().numpy(), [
            fn for fn in test_data["image_meta_dict"]["filename_or_obj"]]


if args.model.tabular_data and args.model.name not in ["DAFT", "FILM", "INTERACTIVE", "HeterogeneousResNet",
                                                       "ConcatHNN1FC", "ConcatHNN2FC",
                                                       "ConcatHNNMCM", "TABATTENTION", "TABMIXER"]:
    if args.model.concat_tabular and args.model.tabular_module != "":
        raise ValueError(
            f"concat_tabular set to True and tabular module is {args.model.tabular_module}! Change tabular module to \"\" or switch off concat_tabular")
    if args.model.concat_tabular:
        tabular_config = None
        tab_concat_dim = tdl.num_tab
        args.model.tabular_module = "Concat"
    else:
        if args.model.tabular_module == "TabMixer":
            additional_args = {
                "use_tabular_data": args.model.tab_mixer.use_tabular_data,
                "spatial_first": args.model.tab_mixer.spatial_first,
                "use_spatial": args.model.tab_mixer.use_spatial,
                "use_temporal": args.model.tab_mixer.use_temporal,
                "use_channel": args.model.tab_mixer.use_channel,
            }
        else:
            additional_args = {}
        tabular_config = get_tabular_config([args.data.batch_size, 1, *args.data.augmentation.resize_size],
                                            args.model.name,
                                            args.model.tabular_module, tdl.num_tab, additional_args=additional_args)
        tab_concat_dim = None
        if not args.model.multiple_tabular_modules:
            tabular_config = [tabular_config[-1]]
else:
    tabular_config = None
    tab_concat_dim = None

print("--------------------")
for fold, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
    print(f"FOLD {fold}")
    print("-------------------")

    train_subsampler = RandomSampler(train_dataset)
    val_subsampler = RandomSampler(val_dataset)
    test_subsampler = RandomSampler(test_dataset)

    train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1,
                                    batch_size=args.data.batch_size, sampler=train_subsampler)
    val_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1,
                                  batch_size=args.data.batch_size_validation, sampler=val_subsampler)

    test_loader = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1,
                                   batch_size=args.data.batch_size_validation, sampler=test_subsampler)

    if args.model.name == "SwinTransformer":
        model = SwinTransformer3D(in_chans=1, last_layer_bias=args.data.last_layer_bias, tabular_config=tabular_config,
                                  tab_concat_dim=tab_concat_dim)
    elif args.model.name == "ResNet18":
        model = r3d_18(last_layer_bias=args.data.last_layer_bias, tabular_config=tabular_config,
                       tab_concat_dim=tab_concat_dim)
    elif args.model.name == "Inception":
        model = InceptionI3d(in_channels=1, num_classes=1, last_layer_bias=args.data.last_layer_bias,
                             tabular_config=tabular_config, tab_concat_dim=tab_concat_dim)
    elif args.model.name == "DAFT":
        model = DAFT_MODEL(in_channels=1, n_outputs=1, filmblock_args={"ndim_non_img": tdl.num_tab},
                           last_layer_bias=args.data.last_layer_bias)
    elif args.model.name == "FILM":
        model = FILM_MODEL(in_channels=1, n_outputs=1, filmblock_args={"ndim_non_img": tdl.num_tab},
                           last_layer_bias=args.data.last_layer_bias)
    elif args.model.name == "TABATTENTION":
        model = TABATTENTION_MODEL(in_channels=1, n_outputs=1, filmblock_args={"ndim_non_img": tdl.num_tab})
    elif args.model.name == "INTERACTIVE":
        model = INTERACTIVE_MODEL(in_channels=1, n_outputs=1, ndim_non_img=tdl.num_tab,
                                  last_layer_bias=args.data.last_layer_bias)
    elif args.model.name == "TABMIXER":
        model = TABMIXER_MODEL(in_channels=1, n_outputs=1, filmblock_args={"ndim_non_img": tdl.num_tab})
    elif args.model.name == "HeterogeneousResNet":
        model = HeterogeneousResNet(in_channels=1, n_outputs=1)
    elif args.model.name == "ConcatHNN1FC":
        model = ConcatHNN1FC(in_channels=1, n_outputs=1, ndim_non_img=tdl.num_tab)
    elif args.model.name == "ConcatHNN2FC":
        model = ConcatHNN2FC(in_channels=1, n_outputs=1, ndim_non_img=tdl.num_tab)
    elif args.model.name == "ConcatHNNMCM":
        model = ConcatHNNMCM(in_channels=1, n_outputs=1, ndim_non_img=tdl.num_tab)
    elif args.model.name == "MLP3D":
        model = MLP3D_T(in_chans=1, num_classes=1, mixing_type=args.model.mlp3d_mixing_type,
                        tabular_config=tabular_config, tab_concat_dim=tab_concat_dim)
    else:
        raise NotImplementedError(f"There is no implementation of: {args.model.name}")

    model = model.to(device)

    # Optimizer
    if args.optimizer.name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay,
                              momentum=0.9)
    elif args.optimizer.name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay,
                                eps=args.optimizer.adam_eps)
    else:
        raise NotImplementedError(f"There are no implementation of: {args.optimizer}")

    if args.model.continue_training:
        model.load_state_dict(torch.load(args.model.trained_model, map_location=device)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.model.rained_model, map_location=device)['optimizer_state_dict'])
        args.start_epoch = torch.load(args.model.trained_model)['epoch']
        print(f'Loaded model, optimizer, starting with epoch: {args.start_epoch}')

    if args.optimizer.scheduler_name == 'annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.intervals.epochs, verbose=True)
    elif args.optimizer.scheduler_name == 'warmup':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.optimizer.warmup_steps, warmup_multiplier=0.01,
                                         t_total=args.intervals.epochs, verbose=False)

    # METRICS
    reduction = 'mean_batch'
    metrics = [MAEMetric(reduction=reduction), RMSEMetric(reduction=reduction), MSEMetric(reduction=reduction)]

    # RUNNING_AVERAGES
    # training loss
    train_loss_cum = CumulativeAverage()
    training_loss_cms = [train_loss_cum]
    # training metrics
    train_mae_cum = CumulativeAverage()
    train_rmse_cum = CumulativeAverage()
    train_mse_cum = CumulativeAverage()
    training_metrics_cms = [train_mae_cum, train_rmse_cum, train_mse_cum]
    # validation metrics
    val_mae_cum = CumulativeAverage()
    val_rmse_cum = CumulativeAverage()
    val_mse_cum = CumulativeAverage()
    val_metrics_cms = [val_mae_cum, val_rmse_cum, val_mse_cum]
    # test metrics
    test_mae_cum = CumulativeAverage()
    test_rmse_cum = CumulativeAverage()
    test_mse_cum = CumulativeAverage()
    test_metrics_cms = [test_mae_cum, test_rmse_cum, test_mse_cum]

    best_score = 999999.99
    best_val_score = 99999.999
    test_preds = {"y_pred": [], "y": [], "pids": []}
    for epoch in range(args.intervals.start_epoch, args.intervals.epochs):
        start_time_epoch = time.time()
        print(f"Starting epoch {epoch + 1}")

        epoch_time = 0.0

        model.train()
        for batch_idx, train_data in enumerate(train_loader):
            training_step(batch_idx, train_data, args)

        epoch_time = time.time() - start_time_epoch

        # RESET METRICS for training
        _ = [func.reset() for func in metrics]

        # VALIDATION

        model.eval()
        with torch.no_grad():
            if (epoch + 1) % args.intervals.validation == 0:
                print("Starting validation...")
                start_time_validation = time.time()
                for batch_idx, val_data in enumerate(val_loader):
                    validation_step(batch_idx, val_data, args)
                val_time = time.time() - start_time_validation
                print(f"Validation time: {val_time:.2f}s")

            # RESET METRICS for validation
            _ = [func.reset() for func in metrics]

            # AGGREGATE RUNNING AVERAGES
            train_loss_agg = [cum.aggregate() for cum in training_loss_cms]
            train_metrics_agg = [cum.aggregate() for cum in training_metrics_cms]
            val_metrics_agg = [cum.aggregate() for cum in val_metrics_cms]

            if best_val_score > val_metrics_agg[0]:
                print("New best validation; starting testing...")
                test_preds = {"y_pred": [], "y": [], "pids": []}
                start_time_test = time.time()
                for batch_idx, test_data in enumerate(test_loader):
                    pred, gt, pid = test_step(batch_idx, test_data, args)
                    test_preds["y_pred"].extend(pred)
                    test_preds["y"].extend(gt)
                    test_preds["pids"].extend(pid)
                test_time = time.time() - start_time_test

                print(f"Testing time: {test_time:.2f}s")
            test_metrics_agg = [cum.aggregate() for cum in test_metrics_cms]

            # RESET METRICS for testing
            _ = [func.reset() for func in metrics]

            # reset running averages
            _ = [cum.reset() for cum in training_loss_cms]
            _ = [cum.reset() for cum in training_metrics_cms]
            _ = [cum.reset() for cum in val_metrics_cms]
            _ = [cum.reset() for cum in test_metrics_cms]

            scheduler.step()

            # # CHECKPOINTS SAVE
            directory = args.model.checkpoint_dir
            if not os.path.exists(directory):
                os.makedirs(directory)

            # save best TRAIN model
            if best_score > train_metrics_agg[0]:
                if args.model.tabular_data:
                    save_path = f"{directory}/model-{global_args.project_name}-{args.model.name}-{args.model.tabular_module}-m{args.model.multiple_tabular_modules}-fold-{fold}_current_best_train.pt"
                else:
                    save_path = f"{directory}/model-{global_args.project_name}-{args.model.name}-fold-{fold}_current_best_train.pt"

                torch.save({
                    'epoch': (epoch),
                    'model_state_dict': model.state_dict(),
                    'model_val_mae': train_metrics_agg[0],
                    'model_val_rmse': train_metrics_agg[1]
                }, save_path)
                best_score = train_metrics_agg[0]
                print(f"Current best train mae score {best_score:.4f}. Model saved!")

            # save best VALIDATION score
            if best_val_score > val_metrics_agg[0]:
                if args.model.tabular_data:
                    if args.model.tabular_module == "TabMixer":
                        save_path = f"{directory}/model-{global_args.project_name}-{args.model.name}-{args.model.tabular_module}-m{args.model.multiple_tabular_modules}-t{args.model.tab_mixer.use_tabular_data}-sf{args.model.tab_mixer.spatial_first}-s{args.model.tab_mixer.use_spatial}-t{args.model.tab_mixer.use_temporal}-fold-{fold}_current_best_val.pt"
                    else:
                        save_path = f"{directory}/model-{global_args.project_name}-{args.model.name}-{args.model.tabular_module}-m{args.model.multiple_tabular_modules}-fold-{fold}_current_best_val.pt"
                else:
                    save_path = f"{directory}/model-{global_args.project_name}-{args.model.name}-fold-{fold}_current_best_val.pt"
                torch.save({
                    'epoch': (epoch),
                    'model_state_dict': model.state_dict(),
                    'model_val_mae': val_metrics_agg[0],
                    'model_val_rmse': val_metrics_agg[1]
                }, save_path)
                best_val_score = val_metrics_agg[0]
                print(f"Current best validation mae {best_val_score:.4f}. Model saved!")

        print(
            f"Epoch: {epoch + 1} finished. Total training loss: {train_loss_agg[0]:.4f} - total epoch time: {epoch_time:.2f}s.")

    df_preds = pd.DataFrame.from_dict(test_preds)
    df_preds["fold"] = fold
    if global_args.project_name == "ph_4ch" or global_args.project_name == "ph_sa":
        total_model_metrics, by_fold_metrics, by_bins_metrics = compute_metrics(df_preds,
                                                                                model_name=args.model.name)

    if args.model.tabular_data:
        total_model_metrics["TABULAR_MODULE"] = args.model.tabular_module
        if args.model.multiple_tabular_modules:
            total_model_metrics["MULTIPLE_MODULES"] = "multiple"
        else:
            total_model_metrics["MULTIPLE_MODULES"] = "single"
    else:
        total_model_metrics["MULTIPLE_MODULES"] = ""
        total_model_metrics["TABULAR_MODULE"] = ""
    print(f"Training finished!")
