from os.path import join as pjoin

def write_done_txt(path, out_name="DONE"):
    with open(pjoin(path, out_name), "w") as done:
        done.write("DONE")