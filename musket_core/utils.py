import yaml
import pickle
import os

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f);

def save_yaml(path,data):
    with open(path, "w") as f:
        return yaml.dump(data,f)

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f);

def save(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

def ensure(directory):
    try:
        os.makedirs(directory);
    except:
        pass

def readArray(_arr, pathPrefix:str, ext:str, expextedSize=None):

    list = _arr if isinstance(_arr, list) else [_arr]

    if not ext.startswith("."):
        ext = "." + ext

    ind = 0
    for arr in list:
        arrInd = 0
        while arrInd < len(arr) and os.path.exists(pathPrefix + str(ind) + ext):
            y = load(pathPrefix + str(ind) + ext)
            l = y.shape[0]
            arr[arrInd:arrInd + l] = y
            arrInd = arrInd + l
            ind = ind + 1

    if expextedSize is not None and expextedSize != arrInd:
        raise ValueError(f"Expected size {expextedSize} bot got {arrInd}")

def dumpArray(_arr, pathPrefix:str, ext:str, blockSize =1024 * 1024 * 512):

    list = _arr if isinstance(_arr, list) else [ _arr ]

    if not ext.startswith("."):
        ext = "." + ext

    ind = 0
    for arr in list:
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
