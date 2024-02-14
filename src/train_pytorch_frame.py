import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_frame import numerical, categorical
from torch_frame import stype
from torch_frame.data import DataLoader, Dataset
from torch_frame.nn import (
    EmbeddingEncoder,
    FTTransformer,
    LinearBucketEncoder,
    LinearEncoder,
    LinearPeriodicEncoder,
    ResNet,
    Trompt,
)
from cuda import setup_cuda
from torch.optim.lr_scheduler import CosineAnnealingLR
from metrics import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="ph")
parser.add_argument('--tabular_data_path', type=str, default="")
parser.add_argument('--numerical_encoder_type', type=str, default='linear',
                    choices=['linear', 'linearbucket', 'linearperiodic'])
parser.add_argument('--model_type', type=str, default='trompt_pf',
                    choices=['trompt_pf', 'fttransformer_pf', 'resnet_pf'])
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu_frac', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=700)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()

log_comet = True

setup_cuda(args.gpu_frac, num_threads=8, device="cuda", visible_devices="0,1", use_cuda_with_id=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=1)
if device.type == "cuda":
    device_name = torch.cuda.get_device_name(int(1))


def extract_pid_from_path(path):
    pid = path.split("/")[-2]
    return pid


if "ph" in args.project_name:
    col_to_stype = {} # TODO

df = pd.read_csv(args.tabular_data_path, index_col=0)
split_info_df = df["test_set_fold"]
df = df.drop(columns=["test_set_fold"])
df_train = df[split_info_df == 1]
df_val = df[split_info_df == 0]
df_test = df[split_info_df == -1]
train_dataset = Dataset(df_train, col_to_stype=col_to_stype, target_col="y")
val_dataset = Dataset(df_val, col_to_stype=col_to_stype, target_col="y")
test_dataset = Dataset(df_test, col_to_stype=col_to_stype, target_col="y")
train_dataset.materialize()
val_dataset.materialize()
test_dataset.materialize()

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

torch.manual_seed(args.seed)

if args.numerical_encoder_type == 'linear':
    numerical_encoder = LinearEncoder()
elif args.numerical_encoder_type == 'linearbucket':
    numerical_encoder = LinearBucketEncoder()
elif args.numerical_encoder_type == 'linearperiodic':
    numerical_encoder = LinearPeriodicEncoder()
else:
    raise ValueError(
        f'Unsupported encoder type: {args.numerical_encoder_type}')

stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: numerical_encoder,
}

output_channels = 1

if args.model_type == 'fttransformer_pf':
    model = FTTransformer(
        channels=args.channels,
        out_channels=output_channels,
        num_layers=args.num_layers,
        col_stats=train_dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    ).to(device)
elif args.model_type == 'resnet_pf':
    model = ResNet(
        channels=args.channels,
        out_channels=output_channels,
        num_layers=args.num_layers,
        col_stats=train_dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
    ).to(device)
elif args.model_type == "trompt_pf":
    model = Trompt(
        channels=args.channels,
        out_channels=output_channels,
        num_prompts=128,
        num_layers=6,
        col_stats=train_dataset.col_stats,
        col_names_dict=train_tensor_frame.col_names_dict,
    ).to(device)
else:
    raise ValueError(f'Unsupported model type: {args.model_type}')

model = torch.compile(model, dynamic=True) if args.compile else model
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=100, verbose=True)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        pred = model(tf)
        loss = F.mse_loss(pred.view(-1), tf.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    scheduler.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader, return_preds=False) -> float:
    model.eval()
    accum = total_count = 0
    if return_preds:
        preds = []
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if return_preds:
            preds.extend(pred.detach().cpu().numpy().flatten())
        accum += float(
            F.mse_loss(pred.view(-1), tf.y.view(-1), reduction='sum'))
        total_count += len(tf.y)

    rmse = (accum / total_count) ** 0.5
    if not return_preds:
        return rmse
    else:
        return rmse, preds


metric = 'RMSE'
best_val_metric = float('inf')
best_test_metric = float('inf')
best_preds = {"y": df_test["y"], "y_pred": [], "pids": list(df_test.index)}

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)

    if val_metric < best_val_metric:
        test_metric, preds = test(test_loader, return_preds=True)
        best_preds["y_pred"] = preds
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
          f'Val {metric}: {val_metric:.4f}, Test {metric}: {best_test_metric:.4f}')

df_preds = pd.DataFrame.from_dict(best_preds)
df_preds["fold"] = 0
if "ph" in args.project_name:
    total_model_metrics, by_fold_metrics, by_bins_metrics = compute_metrics(df_preds,
                                                                            model_name=args.model_type)
print(f'Best Val {metric}: {best_val_metric:.4f}, '
      f'Best Test {metric}: {best_test_metric:.4f}')
