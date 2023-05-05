import sys

sys.path.append("../")
sys.path.append("/../..")

import os
import pickle
import shutil

from time import time

import pandas as pd
import sklearn.metrics

from utils.write_done import write_done_txt
from utils.helpers import fetch_dataset_as_df, get_project_root, get_saved_or_newly_generated_seed
from tpot import TPOTClassifier

from os.path import join as pjoin
from os import getcwd

from utils.io_funcs import load_json
import argparse
from deap import gp



def fetch_and_split_data_set(ds):
    train, test = fetch_dataset_as_df(ds)
    X_train, X_test = train.drop("target", axis=1), test.drop("target", axis=1)
    y_train, y_test = train[["target"]], test[["target"]],
    return X_train.values, X_test.values, y_train.values.ravel(), y_test.values.ravel()


def handle_run_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)


def save_incumbent(automl, save_to):
    automl.export(pjoin(save_to, "incumbent.py"))
    inc = automl._optimized_pipeline

    individual_summary = [gp.graph(inc), inc.fitness.values[-1]]

    with open(pjoin(save_to, "incumbent.pkl"), "wb") as outfile:
        pickle.dump(individual_summary, outfile)


def predict_and_save(aml, x_test, save_to_dir):
    pred = aml.predict(x_test)
    pred_df = pd.DataFrame()
    pred_df["pred"] = pred

    try:
        pred_proba = aml.predict_proba(x_test)
        for i in range(0, pred_proba.shape[1]):
            pred_df[f"pred_proba_{i}"] = pred_proba[:, i]
    except RuntimeError:
        print("this pipeline doesnt have predict_proba")
        pred_df["pred_proba"] = ["Not available for this pipeline"] * pred_df.shape[0]

    pred_df.to_csv(pjoin(save_to_dir, "prediction.csv"), index=False)

    return pred_df


def score_and_save_perf(aml, pred_df, y_test, opt_time, save_to_dir):
    scores_df = pd.DataFrame()
    scores_df["training_score"] = aml._optimized_pipeline.fitness.values[-1]
    scores_df["testing_Score"] = sklearn.metrics.accuracy_score(y_test, pred_df["pred"].values.ravel())
    scores_df["opt_time"] = opt_time

    scores_df.to_csv(pjoin(save_to_dir, "stats.csv"), index=False)


def main():
    automl_params = load_json(args.automl_params_path)

    P_ROOT = get_project_root()
    DS = pjoin(P_ROOT, pjoin("datasets", args.dataset))
    CWD = getcwd()
    SEED = get_saved_or_newly_generated_seed()
    RUN_DIR = pjoin(CWD, f"tpot-ds_{args.dataset}-seed_{SEED}")
    POPS = pjoin(RUN_DIR, "pops")

    handle_run_dir(RUN_DIR)
    handle_run_dir(POPS)

    automl_params["periodic_checkpoint_folder"] = RUN_DIR
    automl_params["log_file"] = pjoin(RUN_DIR, "log.txt")
    automl_params["random_state"] = SEED

    X_train, X_test, y_train, y_test = fetch_and_split_data_set(DS)

    automl = TPOTClassifier(log_file=pjoin(RUN_DIR, "log.txt"),
                            periodic_checkpoint_folder=RUN_DIR,
                            random_state=SEED,
                            verbosity=automl_params["verbosity"],
                            generations=automl_params["generations"],
                            population_size=automl_params["population_size"],
                            max_time_mins=automl_params["max_time_mins"],
                            max_eval_time_mins=automl_params["max_eval_time_mins"],
                            memory=automl_params["memory"]

                            )

    tick = time()
    automl.fit(X_train, y_train)
    tock = time()
    total_time = tock - tick
    print(f"TPOT took {total_time}")

    print()
    save_incumbent(automl, RUN_DIR)
    predictions = predict_and_save(automl, X_test, RUN_DIR)

    score_and_save_perf(automl, predictions, y_test, total_time, RUN_DIR)
    write_done_txt(RUN_DIR)
    print("DONE")
    print("===================================================================== \n\n\n")


if __name__ == "__main__":

    P_ROOT = get_project_root()
    parser = argparse.ArgumentParser(description='Run TPOT')
    parser.add_argument("--automl_params_path", type=str, default=pjoin(P_ROOT, "config/tpot_params.json"))
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    for d in [
        "vehicle"]:#, "higgs", "Amazon_employee_access"]:  # , "KDDCup09-Appetency", "APSFailure", "volkert", "covertype"]:
        args.dataset = d
        print(args.dataset)
        try:
            main()
        except:
            print(f"Error {d}")
