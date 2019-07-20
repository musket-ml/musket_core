import yaml
import pickle
import os
from threading import Lock


_l=Lock()


def load_yaml(path):
    _l.acquire()
    try:
        with open(path, "r") as f:
            return yaml.load(f);
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
