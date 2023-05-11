import sys
import time

sys.path.append("../")
import os
from os import getcwd
from os.path import join as pjoin
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.helpers import get_saved_or_newly_generated_seed, get_project_root
import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

SEED = get_saved_or_newly_generated_seed()
ROOT_DIR = get_project_root()

CWD = pjoin(ROOT_DIR, getcwd())
DATASETS_DIR = pjoin(ROOT_DIR, "datasets")

datasets_ids = [
    # binary classification

    # 9911, #amazon_employee_access
    219,  # electricity
    168868, #APSFailure, # with missing
    3917, #kc1


    # multi class
    9986,  # gas-drift
    45, #splice
    168331,  #volkert

]


def fetch_task(d_id):
    task = openml.tasks.get_task(d_id)
    X, y = task.get_X_and_y(dataset_format='dataframe')
    d_name = task.get_dataset().name

    return openml.tasks.get_task(d_id), X, y, d_name



def encode_labels(y):
    le = LabelEncoder()
    classes = list(y.unique())
    le.fit(classes)
    return le.transform(y)


def split_train_test(X, y, task, target_name="target"):
    train_indices, test_indices = task.get_train_test_split_indices(fold=1)

    data = X
    data[target_name] = y

    train = data.loc[train_indices]
    test = data.loc[test_indices]

    return train, test


def get_dataset_metadata(task):
    dataset_id = task.dataset_id
    all_meta_data = openml.datasets.list_datasets(output_format="dataframe")
    sliced = all_meta_data[all_meta_data["did"] == dataset_id]
    return sliced


def drop_unnamed_col(df):
    unnamed_col = "Unnamed: 0"

    if unnamed_col in list(df.columns):
        return df.drop([unnamed_col], axis=1)
    return df


if __name__ == "__main__":

    if os.path.exists(DATASETS_DIR) is False:
        os.mkdir(DATASETS_DIR)

    metadata_rows = []
    for d in datasets_ids:
        print("fetching task id", d)
        tick = time.time()
        task, X, y, name = fetch_task(d)
        print(task.dataset_id)
        tock = time.time()
        download_mins = (tock - tick) / 60

        print(f"fetched dataset {name} in {int(download_mins)} minutes")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        y = encode_labels(y)
        print("encoded")
        train, test = split_train_test(X, y, task)
        print("split")
        metadata = get_dataset_metadata(task)
        print("metadata acquired")
        dataset_dir = pjoin(DATASETS_DIR, name)

        if os.path.exists(dataset_dir) is False:
            os.mkdir(dataset_dir)
            print("dataset dir created")

        if "Unnamed: 0" in list(train.columns):
            train = train.drop("Unnamed: 0", axis=1)

        train = drop_unnamed_col(train)
        test = drop_unnamed_col(test)

        train.to_csv(pjoin(dataset_dir, "train.csv"), index=False)
        test.to_csv(pjoin(dataset_dir, "test.csv"), index=False)
        metadata.to_csv(pjoin(dataset_dir, "metadata.csv"), index=False)
        print("saved!")
        print("=================")
        print("\n\n")
