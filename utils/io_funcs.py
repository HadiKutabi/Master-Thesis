import pickle
import json

def write_txt(string: str or object, path: object) -> None:
    if not isinstance(string, str):
        string = str(string)

    with open(path, "w+") as f:
        f.write(string)


def read_txt(path: str) -> None:
    with open(path, "r") as f:
        return f.read()


def dump_pickle(obj: object, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: object) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
