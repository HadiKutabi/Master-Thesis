import os
import shutil
from helpers import get_project_root

RUNS_OUTPUT_DIR = os.path.join(get_project_root(), "automl_outputs")

def make_run_dirs(dir_name):

    if os.path.exists(RUNS_OUTPUT_DIR) is False:
        os.mkdir(RUNS_OUTPUT_DIR)

    full_path = os.path.join(RUNS_OUTPUT_DIR, dir_name)

    if os.path.exists(full_path) is True:
        print(f"DIR {full_path} exists")
        print("DELETING DIR")
        shutil.rmtree(full_path)

    os.mkdir(full_path)
    print(f"Made path {full_path}")