import os
import sys

import yaml

import zipfile

import urllib

from urllib.parse import urlparse

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.api_client import ApiClient
except:
    print("can't import kaggle!")

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

    urllib.request.urlretrieve(url, file_dest)

    for item in os.listdir(dest):
        path = os.path.join(dest, item)

        if zipfile.is_zipfile(path):
            new_dir = path[0: path.rindex(".")]

            with zipfile.ZipFile(path) as zip:
                zip.extractall(new_dir)

                print("removing: " + path)

            os.remove(path)

    print("download complete")

def downloadDataset(id, dest):
    api = KaggleApi(ApiClient())

    api.authenticate()

    api.dataset_download_files(id, dest, quiet=False, unzip=True)

    print("download complete")


def downloadCompetitionFiles(id, dest):
    api = KaggleApi(ApiClient())

    api.authenticate()

    api.competition_download_files(id, dest, True, False)

    for item in os.listdir(dest):
        path = os.path.join(dest, item)

        if zipfile.is_zipfile(path):
            new_dir = path[0: path.rindex(".")]

            with zipfile.ZipFile(path) as zip:
                zip.extractall(new_dir)

                print("removing: " + path)

            os.remove(path)

    print("download complete")

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
        return Loader(lambda id, dest: downloadDataset(id, dest))

    if parsed_url["type"] == KAGGLE_COMPETITION:
        return Loader(lambda id, dest: downloadCompetitionFiles(id, dest))

    if parsed_url["type"] == HTTP:
        return Loader(lambda id, dest: download_url(id, dest))

def load_item(url, dest):
    parsed = parse_url(url)

    loader = build_loader(parsed)

    loader.load(parsed["url"], dest)

def download(root):
    full_path = os.path.join(root, "deps.yaml")

    deps = load_yaml(full_path)["dependencies"]

    for item in deps:
        load_item(item, os.path.join(root, "data"))

def main(*args):
    root = args[0][1]

    download(root)

if __name__ == '__main__':
    main(sys.argv)