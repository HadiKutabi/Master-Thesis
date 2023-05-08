import sys

sys.path.append("../")
sys.path.append("/../..")

from alphad3m_containers import DockerAutoML

# from alphad3m import AutoML

from sklearn.metrics import accuracy_score
import pandas as pd
from utils.write_done import write_done_txt
from utils.helpers import get_project_root, get_saved_or_newly_generated_seed
from utils.io_funcs import load_json, write_txt

import pickle
import argparse
from os.path import join as pjoin
from os import getcwd
from time import time
import os


def get_classification_keywords(ds, target="target"):
    ds_df = pd.read_csv(ds)
    n_classes = ds_df[target].nunique()

    if n_classes > 2:
        return ['classification', 'multiClass', 'tabular']
    elif n_classes == 2:
        return ['classification', 'binary', 'tabular']


def get_and_save_leaderboard(automl, save_to_dir):
    leaderboard = automl.plot_leaderboard().data
    leaderboard.to_csv(pjoin(save_to_dir, "leaderboard"), index=False)
    return leaderboard


def get_and_save_incumbent_id(automl, save_to_dir):
    inc_id = automl.get_best_pipeline_id()

    with open(pjoin(save_to_dir, "inc_id.pkl"), "wb") as outfile:
        pickle.dump(inc_id, outfile)

    return inc_id


def get_and_save_primitives(automl, save_to_dir):
    primitives = automl.ta3.list_primitives()

    prims_df = pd.DataFrame()
    prims_df["primitives"] = primitives
    prims_df.to_csv(pjoin(save_to_dir, "primitives.csv"), index=False)

    return prims_df


def get_and_save_pipeline_profiler_output(automl, save_to_dir):
    pipeline_profiler_input = automl.create_pipelineprofiler_inputs()

    with open(pjoin(save_to_dir, "pipeline_profiler_input.pkl"), "wb") as outfile:
        pickle.dump(pipeline_profiler_input, outfile)

    return pipeline_profiler_input


def train_pipeline(automl, p_id):
    automl.train(p_id)


def predict_pipeline(automl, pipeline_id, test_ds_path, save_to_dir):
    preds = automl.test(pipeline_id=pipeline_id, test_dataset=test_ds_path)

    pred_proba = automl.test(pipeline_id=pipeline_id, test_dataset=test_ds_path, calculate_confidence=True)
    pred_proba["target"] = pred_proba["target"].map(lambda x: f"pred_proba_{x}")

    pivoted_pred_proba = pd.pivot_table(pred_proba, values='confidence', index=['d3mIndex'],
                                        columns=['target'])

    joined = preds.set_index('d3mIndex').join(pivoted_pred_proba)

    joined.to_csv(pjoin(save_to_dir, "prediction.csv"), index=False)

    return joined


def score_and_save_performance(preds, lb, test_ds_path, save_to_dir):
    test_ds = pd.read_csv(test_ds_path)

    train_acc = lb[lb["ranking"] == 1]["accuracy"].values[0]
    test_acc = accuracy_score(test_ds["target"].values.ravel(), preds["target"].values.ravel())

    perf_df = pd.DataFrame()

    perf_df["id"] = [lb[lb["ranking"] == 1]["id"].values[0]]
    perf_df["train_acc"] = [train_acc]
    perf_df["test_acc"] = [test_acc]

    perf_df.to_csv(pjoin(save_to_dir, "performance.csv"), index=False)

    return perf_df


def main():
    automl_params = load_json(args.automl_params_path)

    P_ROOT = get_project_root()
    DS = pjoin(P_ROOT, pjoin("datasets", args.dataset))
    TRAIN = pjoin(DS, "train.csv")
    TEST = pjoin(DS, "test.csv")
    SEED = get_saved_or_newly_generated_seed()
    # RUN_DIR = pjoin(CWD, f"alphad3m-ds_{args.dataset}-seed_{SEED}")

    RUN_DIR = pjoin(pjoin(P_ROOT, "automl_outputs"), f"alphad3m-ds_{args.dataset}-seed_{SEED}")

    automl_params["random_seed"] = SEED

    print()

    automl = DockerAutoML(RUN_DIR, verbose=True)
    tick = time()
    automl.search_pipelines(TRAIN,
                            time_bound=automl_params["time_bound"],
                            time_bound_run=automl_params["time_bound_run"],
                            target=automl_params["target"],
                            metric=automl_params["metric"],
                            task_keywords=get_classification_keywords(TRAIN),
                            random_seed=automl_params["random_seed"])
    tock = time()
    total_time = tock - tick
    print(f"AlphaD3M took {total_time}")


    write_txt(str(total_time), pjoin(RUN_DIR, "total_time.txt"))
    print("Saved total time")

    lb = get_and_save_leaderboard(automl, RUN_DIR)
    print("Saved leaderboard")

    prims = get_and_save_primitives(automl, RUN_DIR)
    print("Saved leaderboard")

    incumbent = get_and_save_incumbent_id(automl, RUN_DIR)
    print("Saved incumbent")

    pipeline_profiler_input = get_and_save_pipeline_profiler_output(automl, RUN_DIR)
    print("Saved PipelineProfiler input list")

    train_pipeline(automl, incumbent)
    preds = predict_pipeline(automl, incumbent, TEST, RUN_DIR)
    print("Saved predictions")

    performance = score_and_save_performance(preds, lb, TEST, RUN_DIR)
    print("Saved scores")

    write_done_txt(RUN_DIR)
    automl.end_session()
    print("DONE")
    print("===================================================================== \n\n\n")

    # automl.export_pipeline_code()




if __name__ == "__main__":

    P_ROOT = get_project_root()
    parser = argparse.ArgumentParser(description='Run AlphaD3M')
    parser.add_argument("--automl_params_path", type=str, default=pjoin(P_ROOT, "config/alphad3m_params.json"))
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    for d in os.listdir(pjoin(P_ROOT, "datasets")):

        args.dataset = d
        print(args.dataset)
        try:
            main()
        except Exception as e:
            print(f"Error {args.dataset}")
            print(e)
