import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f);

def save_yaml(path,data):
    with open(path, "w") as f:
        return yaml.dump(data,f)