import yaml
import pickle
import os

import shutil

from threading import Lock


_l=Lock()

def load_yaml(path):
    _l.acquire()
    try:
        yaml_load = lambda x: yaml.load(x, Loader=yaml.Loader)

        with open(path, "r") as f:
            return yaml_load(f);
    finally:
        _l.release()

def load_string(path):
    with open(path, 'r') as myfile:
        data = myfile.read()
        return data
def save_string(path,data):
    with open(path, 'w') as myfile:
        myfile.write(data)

def save_yaml(path, data, header=None):
    _l.acquire()
    try:
        with open(path, "w") as f:
            if header:
                text = yaml.dump(data)

                text = header + "\n" + text;

                f.write(text)

                return None
            return yaml.dump(data, f)
    finally:
        _l.release()

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f);


def save(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

def delete_file(path:str, recursively = True):
    if os.path.isdir(path):
        if recursively:
            for chPath in (os.path.join(path, f) for f in os.listdir(path)):
                delete_file(chPath, True)
        os.rmdir(path)
    else:
        os.unlink(path)


def ensure(directory):
    try:
        os.makedirs(directory);
    except:
        pass

def readArray(_arr, pathPrefix:str, ext:str, message:str,expextedSize=None):

    lst = _arr if isinstance(_arr, list) else [_arr]

    if not ext.startswith("."):
        ext = "." + ext

    ind = 0
    for arr in lst:
        arrInd = 0
        while arrInd < len(arr) and os.path.exists(pathPrefix + str(ind) + ext):
            y = load(pathPrefix + str(ind) + ext)
            l = y.shape[0]
            arr[arrInd:arrInd + l] = y
            arrInd = arrInd + l
            ind = ind + 1

    if expextedSize is not None and expextedSize != arrInd:
        raise ValueError(f"Expected size {expextedSize} bot got {arrInd}")

def dumpArray(_arr, pathPrefix:str, ext:str, message,blockSize =1024 * 1024 * 512):

    lst = _arr if isinstance(_arr, list) else [ _arr ]

    if not ext.startswith("."):
        ext = "." + ext

    ind = 0
    for arr in lst:
        l = len(arr)
        i0 = arr[0]
        itemSize = i0.size * i0.dtype.itemsize
        bufLength = blockSize // itemSize
        if bufLength == 0:
            bufLength = 1

        for i in range(0, l, bufLength):
            end = min(i + bufLength, l)
            save(pathPrefix + str(ind) + ext,arr[i:end])
            ind = ind + 1

def templates_folder():
    root = os.path.dirname(__file__)

    return os.path.join(root, "templates")

def copy_tree(src, dst):
    return shutil.copytree(src, dst, True)

def copy_file(src, dst):
    return shutil.copy(src, dst)

def archive(target_folder, output_path):
    shutil.make_archive(output_path, 'zip', target_folder)

def collect_project(src, dst):
    project_dst = os.path.join(dst, os.path.basename(src))

    ensure(project_dst)

    def filter(item):
        if item == "experiments":
            return False

        if item == "kaggle":
            return False

        if item == "data":
            return False

        if item == ".metadata":
            return False

        if item.startswith('.'):
            return False

        return True

    directories = [item for item in os.listdir(src) if filter(item)]

    for item in directories:
        src_path = os.path.join(src, item)
        dst_path = os.path.join(project_dst, item)

        if os.path.isdir(src_path):
            copy_tree(src_path, dst_path)
        else:
            copy_file(src_path, dst_path)

    experiments = [item for item in os.listdir(os.path.join(src, "experiments")) if not item.startswith('.')]

    for item in experiments:
        src_path = os.path.join(src, "experiments", item, "config.yaml")
        dst_path = os.path.join(project_dst, "experiments", item, "config.yaml")

        ensure(os.path.dirname(dst_path))

        copy_file(src_path, dst_path)