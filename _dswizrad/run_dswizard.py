import logging
import os
import sys

import numpy as np

sys.path.append("../")
sys.path.append("/../..")

from time import time
import pandas as pd
from sklearn.metrics import accuracy_score

import argparse
from os import getcwd
from os.path import join as pjoin

from utils.helpers import get_project_root, get_saved_or_newly_generated_seed, fetch_dataset_as_df
from utils.io_funcs import load_json, dump_pickle

from dswizard.util import util
from dswizard.core.model import Dataset
from dswizard.core.master import Master


def fetch_and_get_dataset_ready(ds):
    train, test = fetch_dataset_as_df(ds)
    X_train, X_test = train.drop("target", axis=1), test.drop("target", axis=1)
    y_train, y_test = train[["target"]], test[["target"]],

    train_ds = Dataset(X_train.values, y_train.values.ravel(), metric="accuracy", task=123, fold=0,
                       feature_names=list(X_train.columns))
    test_ds = Dataset(X_test.values, y_test.values.ravel(), metric="accuracy", task=123, fold=0,
                      feature_names=list(X_train.columns))

    return train_ds, test_ds


# def get_and_save_stats_csv(aml, total_time, accuracy, save_to_dir):
#     _, stats = aml.sprint_statistics()
#     stats = pd.DataFrame(stats)
#     stats["total_time"] = total_time
#     stats["test_score"] = accuracy
#     stats["n_pipelines_in_ensemble"] = len(aml.show_models())
#     stats.to_csv(pjoin(save_to_dir, "run_stats.csv"), index=False)


# def predict_and_save(aml, x_test, save_to_dir):
#     pred = aml.predict(x_test)
#     pred_proba = aml.predict_proba(x_test)
#
#     pred_df = pd.DataFrame()
#
#     pred_df["pred"] = pred
#     pred_df["pred_proba_0"] = pred_proba[:, 0]
#     pred_df["pred_proba_1"] = pred_proba[:, 1]
#
#     pred_df.to_csv(pjoin(save_to_dir, "prediction.csv"), index=False)
#
#     return pred, pred_proba


def make_predictions(pipeline, ensemble, test_ds, save_to_dir):
    preds = pd.DataFrame()

    pipe_pred = pipeline.predict(test_ds.X)
    pipe_pred_proba = pipeline.predict_proba(test_ds.X)
    preds["inc_pred"] = pipe_pred
    for i in range(0, pipe_pred_proba.shape[1]):
        preds[f"inc_pred_proba_{i}"] = pipe_pred_proba[:, i]

    ensemble_pred = ensemble.predict(test_ds.X)
    ensemble_pred_proba = ensemble.predict_proba(test_ds.X)
    preds["ens_pred"] = ensemble_pred
    for i in range(0, ensemble_pred_proba.shape[1]):
        preds[f"ens_pred_proba_{i}"] = ensemble_pred_proba[:, i]

    preds.to_csv(pjoin(save_to_dir, "predictions.csv"), index=False)

    return preds


def compute_and_save_score(pred, test_ds, train_score, save_to_dir):
    df_perf = pd.DataFrame()
    df_perf["train_accuracy"] = train_score
    print(accuracy_score(test_ds.y, pred["inc_pred"].values.ravel()))
    df_perf["incumbent_test_score"] = accuracy_score(test_ds.y, pred["inc_pred"].values.ravel())
    df_perf["ensemble_test_score"] = accuracy_score(test_ds.y, pred["ens_pred"].values.ravel())
    df_perf.to_csv(pjoin(save_to_dir, "scores.csv"), index=False)


def compute_and_save_score(pred, test_ds, train_score, save_to_dir):
    data = {}
    data["train_score"] = [train_score]
    data["incumbent_test_score"] = [accuracy_score(test_ds.y, pred["inc_pred"].values.ravel())]
    data["ensemble_test_score"] = [accuracy_score(test_ds.y, pred["ens_pred"].values.ravel())]

    df_perf = pd.DataFrame.from_dict(data)
    df_perf.to_csv(pjoin(save_to_dir, "scores.csv"), index=False)


def main():
    automl_params = load_json(args.automl_params_path)

    P_ROOT = get_project_root()
    DS = pjoin(P_ROOT, pjoin("datasets", args.dataset))
    CWD = getcwd()
    SEED = get_saved_or_newly_generated_seed()
    RUN_DIR = pjoin(CWD, f"dswizard-ds_{args.dataset}-seed_{SEED}")

    util.setup_logging(os.path.join(RUN_DIR, 'log.txt'))
    logger = logging.getLogger()
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    train_ds, test_ds = fetch_and_get_dataset_ready(DS)

    np.random.seed(SEED)

    automl_params["working_directory"] = RUN_DIR
    automl_params["ds"] = train_ds
    automl_params["logger"] = logger

    print("Starting AutoML procedure")

    automl = Master(**automl_params)
    tick = time()
    pipeline, run_history, ensemble = automl.optimize()
    tock = time()
    total_time = tock - tick

    print(f"DSWIZARD took {total_time}")

    _, incumbent = run_history.get_incumbent()
    dump_pickle(pipeline, pjoin(RUN_DIR, "pipeline.pkl"))
    dump_pickle(ensemble, pjoin(RUN_DIR, "ensemble.pkl"))
    dump_pickle(run_history, pjoin(RUN_DIR, "my_run_history.pkl"))
    dump_pickle(incumbent, pjoin(RUN_DIR, "incumbent.pkl"))

    pred_df = make_predictions(pipeline, ensemble, test_ds, RUN_DIR)
    compute_and_save_score(pred_df, test_ds, -incumbent.get_incumbent().loss, RUN_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Auto-Sklearn1')
    parser.add_argument("--automl_params_path", type=str, default="dswizard_params.json")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    for d in [
        "kc1", "Amazon_employee_access", "higgs", "KDDCup09-Appetency", "APSFailure", "volkert", "covertype"]:
        args.dataset = d
        print(args.dataset)
        main()
