'''
@author: 32kda
'''
import os

def is_experiment(root):
    config_path = os.path.join(root, "config.yaml")

    return os.path.exists(config_path)

def project_path():
    cwd = os.getcwd()

    if is_experiment(cwd):
        return os.path.abspath(os.path.join(cwd, "../../"))

    return cwd