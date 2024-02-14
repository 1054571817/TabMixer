import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def compute_metrics_detailed(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mape, rmse, mse, r2


def compute_stds(y_true, y_pred):
    mae_values = np.abs(y_true - y_pred)
    mae_std = np.std(mae_values)

    ape = np.abs((y_true - y_pred) / y_true)
    mape_std = np.std(ape)

    mse_values = (y_true - y_pred) ** 2
    mse_std = np.std(mse_values)
    return mae_std, mape_std, None, mse_std, None


def compute_by_fold_metrics(df_preds, gt_col="y", pred_col="y_pred"):
    all_metrics = []
    for f in df_preds["fold"].unique():
        y_true = df_preds[df_preds["fold"] == f][gt_col]
        y_pred = df_preds[df_preds["fold"] == f][pred_col]
        metrics = compute_metrics_detailed(y_true, y_pred)
        all_metrics.append([f, *metrics])
    header = ["FOLD", "MAE", "MAPE", "RMSE", "MSE", "R2"]
    metrics = pd.DataFrame(all_metrics, columns=header)
    if df_preds["fold"].nunique() != 1:
        mean = pd.DataFrame(metrics.mean()).T
        mean["FOLD"] = "MEAN"
        std = pd.DataFrame(metrics.std()).T
        std["FOLD"] = "STD"
        result = pd.concat([metrics, mean, std])
    else:
        stds = compute_stds(df_preds[gt_col], df_preds[pred_col])
        stds = pd.DataFrame([["STD", *stds]], columns=header)
        result = pd.concat([metrics, stds])
    return result


def compute_by_bins_metrics(df_preds, gt_col="y", pred_col="y_pred", bins=[20, 25, 30]):
    all_metrics = []
    bins_inf = [-99999, *bins, 99999]
    all_stds = []
    for f in df_preds["fold"].unique():
        for i in range(len(bins_inf) - 1):
            y_true = df_preds[
                (df_preds["fold"] == f) & (df_preds[gt_col] > bins_inf[i]) & (df_preds[gt_col] <= bins_inf[i + 1])][
                gt_col]
            y_pred = df_preds[
                (df_preds["fold"] == f) & (df_preds[gt_col] > bins_inf[i]) & (df_preds[gt_col] <= bins_inf[i + 1])][
                pred_col]
            if len(y_pred) == 0:
                continue
            metrics = compute_metrics_detailed(y_true, y_pred)
            if i == 0:
                bin_text = f"y<={bins_inf[i + 1]}"
            elif i == (len(bins_inf) - 2):
                bin_text = f"{bins_inf[i]}<y"
            else:
                bin_text = f"{bins_inf[i]}<y<={bins_inf[i + 1]}"
            if df_preds["fold"].nunique() == 1:
                stds = compute_stds(y_true, y_pred)
                all_stds.append(["STD", bin_text, *stds])
            all_metrics.append([f, bin_text, *metrics])
    header = ["FOLD", "BIN", "MAE", "MAPE", "RMSE", "MSE", "R2"]
    metrics = pd.DataFrame(all_metrics, columns=header)
    if df_preds["fold"].nunique() != 1:
        mean = metrics.groupby("BIN").mean()
        mean["FOLD"] = "MEAN"
        std = metrics.groupby("BIN").std()
        std["FOLD"] = "STD"
        result = pd.concat([metrics, mean, std])
    else:
        std = pd.DataFrame(all_stds, columns=header)
        result = pd.concat([metrics, std])
    return result


def compute_total_metrics(by_fold_metrics, model_name=""):
    if by_fold_metrics["FOLD"].nunique() == 2:  # only one fold (fold + std)
        mean = by_fold_metrics[by_fold_metrics["FOLD"] == 0]
        std = by_fold_metrics[by_fold_metrics["FOLD"] == "STD"]
    else:
        mean = by_fold_metrics[by_fold_metrics["FOLD"] == "MEAN"]
        std = by_fold_metrics[by_fold_metrics["FOLD"] == "STD"]
    mean.loc[:, "MAPE"] = mean["MAPE"] * 100
    std.loc[:, "MAPE"] = std["MAPE"] * 100
    total_metrics = mean.round(2).astype(str) + "+/-" + std.round(2).astype(str)
    total_metrics["MODEL"] = model_name
    total_metrics = total_metrics.drop(columns=["FOLD"])
    return total_metrics


def compute_metrics(df_preds, gt_col="y", pred_col="y_pred", bins=[20, 25, 30], model_name=""):
    by_fold_metrics = compute_by_fold_metrics(df_preds, gt_col, pred_col)
    by_bins_metrics = compute_by_bins_metrics(df_preds, gt_col, pred_col, bins)
    total_metrics = compute_total_metrics(by_fold_metrics, model_name)
    return total_metrics, by_fold_metrics, by_bins_metrics


def save_metrics(total_metrics, by_fold_metrics, by_bins_metrics, model_name, folder_results):
    total_metrics.to_csv(os.path.join(folder_results, f"{model_name}_total_model_metrics.csv"))
    by_fold_metrics.to_csv(os.path.join(folder_results, f"{model_name}_by_fold_metrics.csv"))
    by_bins_metrics.to_csv(os.path.join(folder_results, f"{model_name}_by_bins_metrics.csv"))
