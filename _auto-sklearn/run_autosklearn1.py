import sys

sys.path.append("../")
sys.path.append("/../..")


from time import time
import autosklearn.classification
import pandas as pd
from sklearn.metrics import accuracy_score

import argparse
from os import getcwd
from os.path import join as pjoin

from utils.helpers import get_project_root, get_saved_or_newly_generated_seed, fetch_dataset_as_df
from utils.io_funcs import load_json, dump_pickle


def get_logging_config():
    return {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "custom": {
                # More format options are available in the official
                # `documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        # Any INFO level msg will be printed to the console
        "handlers": {
            "console": {
                "level": "INFO",
                "formatter": "custom",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "level": "DEBUG",
            },
            "Client-EnsembleBuilder": {
                "level": "DEBUG",
                "handlers": ["console"],
            },
        },
    }


def fetch_and_split_data_set(ds):
    train, test = fetch_dataset_as_df(ds)
    X_train, X_test = train.drop("target", axis=1), test.drop("target", axis=1)
    y_train, y_test = train[["target"]], test[["target"]],
    return X_train, X_test, y_train, y_test


def get_and_save_stats_csv(aml, total_time, accuracy, save_to_dir):
    _, stats = aml.sprint_statistics()
    stats = pd.DataFrame(stats)
    stats["total_time"] = total_time
    stats["test_score"] = accuracy
    stats["n_pipelines_in_ensemble"] = len(aml.show_models())
    stats.to_csv(pjoin(save_to_dir, "run_stats.csv"), index=False)


def predict_and_save(aml, x_test, save_to_dir):
    pred = aml.predict(x_test)
    pred_proba = aml.predict_proba(x_test)

    pred_df = pd.DataFrame()

    pred_df["pred"] = pred
    pred_df["pred_proba_0"] = pred_proba[:, 0]
    pred_df["pred_proba_1"] = pred_proba[:, 1]

    pred_df.to_csv(pjoin(save_to_dir, "prediction.csv"), index=False)

    return pred, pred_proba


def main():
    automl_params = load_json(args.automl_params_path)
    logging_config = get_logging_config()

    P_ROOT = get_project_root()
    DS = pjoin(P_ROOT, pjoin("datasets", args.dataset))
    CWD = getcwd()
    SEED = get_saved_or_newly_generated_seed()
    RUN_DIR = pjoin(CWD, f"autosklearn1-ds_{args.dataset}-seed_{SEED}")

    X_train, X_test, y_train, y_test = fetch_and_split_data_set(DS)

    automl_params["seed"] = SEED
    automl_params["tmp_folder"] = RUN_DIR
    automl_params["logging_config"] = logging_config


    print("Starting AutoML procedure")

    automl = autosklearn.classification.AutoSklearnClassifier(**automl_params)

    tick = time()
    automl.fit(X_train, y_train)
    tock = time()
    total_time = tock - tick
    print(f"AutoSklean took {total_time}")

    dump_pickle(automl, pjoin(RUN_DIR, "autosklearn_obj.pkl"))
    print("saved AutoML object")

    pred, pred_proba = predict_and_save(automl, X_test, RUN_DIR)
    acc = accuracy_score(y_test, pred)
    print("scored")

    get_and_save_stats_csv(automl, total_time, acc, RUN_DIR)
    print("Saved stats")
    print("===================================================================== \n\n\n")

    with open(pjoin(RUN_DIR, "done.txt"), "w+") as done:
        done.write("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Auto-Sklearn1')
    parser.add_argument("--automl_params_path", type=str, default="dswizard_params.json")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    for d in ["kc1"]:#, "Amazon_employee_access", "higgs", "KDDCup09-Appetency", "APSFailure", "volkert", "covertype"]:
        args.dataset = d
        print(args.dataset)
        try:
            main()
        except:
            print("Error")
