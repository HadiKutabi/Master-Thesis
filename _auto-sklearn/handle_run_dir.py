import os
import shutil


def del_run_dir_if_exists(run_dir):
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

        print(f"Removed {run_dir}")