import yaml
import os

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f);

def save_yaml(path,data):
    with open(path, "w") as f:
        return yaml.dump(data,f)

def ensure(directory):
    try:
        os.makedirs(directory);
    except:
        pass