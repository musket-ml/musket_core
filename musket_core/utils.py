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
    with open(path, "r") as f:
        return pickle.load(f);

def save(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

def ensure(directory):
    try:
        os.makedirs(directory);
    except:
        pass