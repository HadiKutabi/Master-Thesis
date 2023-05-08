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
    4532,  # higgs
    4135,  # amazon_employee_access
    54,  # vehicle
    42757,  # KDDCup09-Appetency
    # multi class
    41166,  # volkert
    1596  # covertype

]


def fetch_dataset(d_id):
    dataset = openml.datasets.get_dataset(d_id)
    target_name = dataset.default_target_attribute
    data, _, _, _ = dataset.get_data()

    return data, dataset, target_name


def encode_labels(df, target):
    le = LabelEncoder()
    classes = list(df[target].unique())
    le.fit(classes)
    encoded = le.transform(df[target].tolist())
    df.loc[:, "target"] = encoded
    if target != "target":
        df.drop(target, axis=1, inplace=True)

    return df


def split_train_test(df, test_size=0.25, seed=SEED):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test


def get_dataset_metadata(dataset_id):
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
        print("fetching dataset id", d)
        tick = time.time()
        data, dataset, target_name = fetch_dataset(d)
        tock = time.time()
        mins = int((tock - tick) / 60)
        print(f"fetched dataset {dataset.name} in {mins} minutes")
        print("shape:", data.shape)
        data = encode_labels(data, target_name)
        print("encoded")
        train, test = split_train_test(data, 0.25)
        print("split")
        metadata = get_dataset_metadata(d)
        print("metadata acquired")
        dataset_dir = pjoin(DATASETS_DIR, dataset.name)

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
