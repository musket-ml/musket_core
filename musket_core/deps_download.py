import os
import sys

import yaml

import zipfile

import urllib.request

from musket_core import utils, download_callbacks
from pathlib import Path

from urllib.parse import urlparse
import importlib

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.api_client import ApiClient
except:
    print("Can't import kaggle! Please check kaggle package is installed and ${USER_HOME}/.kaggle/kaggle.json is present.")

KAGGLE_COMPETITION = "kaggle.competition"
KAGGLE_DATASET = "kaggle.dataset"
HTTP = "http"
LOCAL_FILE = "file"

class Loader():
    def __init__(self, loader):
        self.loader = loader

    def load(self, url, dst):
        self.loader(url, dst)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, yaml.Loader)

def download_url(url, dest):
    file_dest = os.path.join(dest, url[url.rindex("/") + 1:])

    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'My User Agent 1.0')
    opener.retrieve(url, file_dest)

    for item in os.listdir(dest):
        path = os.path.join(dest, item)

        if zipfile.is_zipfile(path):
            new_dir = path[0: path.rindex(".")]
            safe_unzip(path, new_dir)
            print("removing: " + path)

            os.remove(path)

    print("download complete")

def downloadDataset(dataset_id, dest):
    api = KaggleApi(ApiClient())

    api.authenticate()

    api.dataset_download_files(dataset_id, dest, quiet=False, unzip=True)

    print("download complete")


def downloadCompetitionFiles(dataset_id, dest):
    api = KaggleApi(ApiClient())

    api.authenticate()

    api.competition_download_files(dataset_id, dest, True, False)

    unpack_all_zips(dest)

    print("download complete")

def unpack_all_zips(dest):
    if not os.path.exists(dest):
        return

    if not os.path.isdir(dest):
        return

    has_zips = True
    while has_zips:
        has_zips = False
        for item in os.listdir(dest):
            path = os.path.join(dest, item)
    
            if zipfile.is_zipfile(path):
                has_zips = True
                safe_unzip(path,dest)
    #             new_dir = path[0: path.rindex(".")]
    # 
    #             with zipfile.ZipFile(path) as zip_file:
    #                 zip_file.extractall(new_dir)
    # 
    #                 print("removing: " + path)
    
                os.remove(path)
                
    for item in os.listdir(dest):
        if os.path.isdir(item):
            unpack_all_zips(item)
            
def safe_unzip(zip_file, dest):
    with zipfile.ZipFile(zip_file, 'r') as zf:
        use_subfolder = False 
        for member in zf.infolist():
            abspath = os.path.abspath(os.path.join(dest, member.filename))
            use_subfolder = use_subfolder or os.path.exists(abspath)
        if not use_subfolder:
            extractpath = dest  
        else:
            extractpath = os.path.join(dest, zf.filename[0: zf.filename.rindex(".")])
            while os.path.exists(extractpath):
                extractpath = extractpath + "1"
        folders = [x for x in zf.infolist() if x.filename.find('/') == len(x.filename) - 1]
        # If archive has a single folder with the similar name inside - extract without extra folder to avoid folder duplication
        if len(folders) == 1 and folders[0].filename[:-1] == os.path.splitext(os.path.basename(extractpath))[0]:
            extractpath = Path(extractpath).parent
        zf.extractall(extractpath)

def parse_url(url):
    if HTTP in url and url.index(HTTP) == 0:
        return {
            "type": HTTP,
            "url": url
        }

    parsed = urlparse(url)

    return {
        "type": parsed.scheme,
        "url": parsed[1] + parsed[2]
    }

def build_loader(parsed_url):
    if parsed_url["type"] == KAGGLE_DATASET:
        return Loader(lambda dataset_id, dest: downloadDataset(dataset_id, dest))

    if parsed_url["type"] == KAGGLE_COMPETITION:
        return Loader(lambda dataset_id, dest: downloadCompetitionFiles(dataset_id, dest))

    if parsed_url["type"] == HTTP:
        return Loader(lambda dataset_id, dest: download_url(dataset_id, dest))

def is_loaded(root, url):
    try:
        fullPath = os.path.join(root, ".metadata", "downloaded_deps.yaml")

        loadedYaml = load_yaml(fullPath)

        return url in loadedYaml["dependencies"]
    except:
        return False


def mark_loaded(root, url):
    fullPath = os.path.join(root, ".metadata")

    utils.ensure(fullPath)

    fullPath = os.path.join(fullPath, "downloaded_deps.yaml")

    try:
        loaded_yaml = load_yaml(fullPath)
    except:
        loaded_yaml = {"dependencies": []}

    deps = loaded_yaml["dependencies"]

    deps.append(url)

    utils.save_yaml(fullPath, loaded_yaml)

def load_item(url, dest):
    utils.ensure(dest)

    parsed = parse_url(url)

    loader = build_loader(parsed)

    loader.load(parsed["url"], dest)

def download(root, force_all=False):
    full_path = os.path.join(root, "project.yaml")

    modules_dir = os.path.join(root, "modules")
    sys.path.insert(0, modules_dir)
    modules = [file[:-3] for file in os.listdir(modules_dir) if file.endswith('.py')]
    for module in modules: #We need this to make decoration happen
        importlib.import_module(module)

    try:
        loadedYaml = load_yaml(full_path)
    except:
        print("no dependencies file found")

        return

    try:
        deps = loadedYaml["dependencies"]
    except:
        print("can not parse project.yaml")

        return

    if not isinstance(deps, list):
        print("can not parse project.yaml")

        return

    data_dir = os.path.join(root, "data")
    force_all = force_all or not os.path.exists(data_dir)

    for item in deps:
        url = item

        data_path = data_dir

        force = False

        if isinstance(item, dict):
            url = item.get("url", False)

            if not url:
                print("incorrect url")

                continue

            dst = item.get("destination", data_path)
            force = item.get("force", False) or not os.path.exists(dst)

            data_path = os.path.abspath(os.path.join(data_path, dst))

            pass

        if is_loaded(root, url) and not (force or force_all):
            print("skipping: " + url)

            continue

        load_item(url, data_path)

        mark_loaded(root, url)
    
    for callback in download_callbacks.get_after_download_callbacks():
        callback()   


def main(*args):       
    idx = args[0].index('--project') if '--project' in args[0] else -1
    if idx >= 0 and idx < len(args[0]) - 1:
        root = args[0][idx + 1]
    elif len(args[0]) > 1 and os.path.exists(args[0][1]):
        root = args[0][1]
    else:
        root = os.getcwd()

    download(root, "--force" in args[0])

if __name__ == '__main__':
    main(sys.argv)

