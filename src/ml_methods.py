import os
import pandas as pd
from metrics import compute_metrics, save_metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def split_folds(df, y_col="y", fold_col="test_set_fold"):
    X_trains = []
    y_trains = []
    y = df[y_col]
    test_set_fold = df[fold_col]
    X = df.drop(columns=[y_col, fold_col])
    X_test = X[test_set_fold == -1]
    y_test = y[test_set_fold == -1]
    X = X[test_set_fold != -1]
    y = y[test_set_fold != -1]
    test_set_fold = test_set_fold[test_set_fold != -1]
    for i in range(test_set_fold.nunique() - 1):  # only one split; remove -1 if there are more splits
        X_trains.append(X[test_set_fold != i])
        y_trains.append(y[test_set_fold != i])
    return X_trains, y_trains, X_test, y_test


def train_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def build_df_fold_predictions(fold, y, predictions):
    return pd.DataFrame(index=y.index, data={"fold": [fold] * len(y), "y": y, "y_pred": predictions})


def compute_ml_methods(path_data, folder_results):
    df = pd.read_csv(path_data, index_col=0)
    X_trains, y_trains, X_test, y_test = split_folds(df)
    models = [LinearRegression, GradientBoostingRegressor, RandomForestRegressor, MLPRegressor, XGBRegressor]
    args = [{}, {}, {}, {}, {}]
    total_results = None
    for model_cls, arg in zip(models, args):
        fold_predictions = pd.DataFrame(columns=["fold", "y", "y_pred"])
        for i, (X_train, y_train) in enumerate(zip(X_trains, y_trains)):
            model = model_cls(**arg)
            predictions = train_predict(model, X_train, y_train, X_test)
            current_fold_predictions = build_df_fold_predictions(i, y_test, predictions)
            if fold_predictions.empty:
                fold_predictions = current_fold_predictions
            else:
                fold_predictions = pd.concat([fold_predictions, current_fold_predictions])
        total_model_metrics, by_fold_metrics, by_bins_metrics = compute_metrics(fold_predictions,
                                                                                model_name=model_cls.__name__,
                                                                                bins=[20, 25, 30])
        model_name = model_cls.__name__
        print(model_name)
        print(total_model_metrics)
        print(by_fold_metrics)
        print(by_bins_metrics)
        print("_______________")
        if total_results is None:
            total_results = total_model_metrics
        else:
            total_results = pd.concat([total_results, total_model_metrics])
        save_metrics(total_model_metrics, by_fold_metrics, by_bins_metrics, model_name, folder_results)
    print(total_results)
    total_results.to_csv(os.path.join(folder_results, "ph_reg_ml_results.csv"), index=False)


if __name__ == '__main__':
    path_data = ""
    folder_results = ""
    compute_ml_methods(path_data, folder_results)
