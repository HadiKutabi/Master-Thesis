import sys

import pandas as pd

sys.path.append("../utils")
from io_funcs import *
from pathlib import Path
from os.path import join as pjoin
from os.path import exists as pexists
import random


#
def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_saved_or_newly_generated_seed(name: str = "seed.txt") -> int:
    full_path = pjoin(get_project_root(), "config")
    full_path = pjoin(full_path, name)

    if pexists(full_path):
        print("A seed exists")
        return int(read_txt(full_path))

    else:
        print("generating and saving a new seed")
        rnd_seed = random.randint(1, 100000)
        write_txt(rnd_seed, full_path)
        return rnd_seed


def fetch_dataset_as_df(name):
    P_ROOT = get_project_root()
    DATASETS_DIR = pjoin(P_ROOT, "datasets")
    dataset_dir = pjoin(DATASETS_DIR, name)
    train = pd.read_csv(pjoin(dataset_dir, "train.csv"))
    test = pd.read_csv(pjoin(dataset_dir, "test.csv"))
    return train, test