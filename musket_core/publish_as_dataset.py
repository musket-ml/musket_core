import os
import shutil

import argparse

from musket_core import utils

import json

def to_dataset(src, experiments, name, data=False):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kaggle.api_client import ApiClient
    except:
        print("Kaggle not found or user credentials not provided.")

        return

    api = KaggleApi(ApiClient())

    api.authenticate()

    dest = os.path.expanduser("~/.musket_core/proj_to_dataset")

    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    utils.ensure(dest)
    
    utils.visit_tree(src, lambda path: utils.throw("zip files not allowed!") if path.lower().endswith(".zip") else ())

    if data:
        src = os.path.join(src, "data")
        dest = os.path.join(dest, "data")

        shutil.copytree(src, dest)
    else:
        utils.collect_project(src, dest, True, False, experiments)

    api.dataset_initialize(dest)

    metapath = os.path.join(dest, "dataset-metadata.json")

    with open(metapath, "r") as f:
        metadata = f.read()

    metadata = metadata.replace("INSERT_SLUG_HERE", name).replace("INSERT_TITLE_HERE", name)

    with open(metapath, "w") as f:
        f.write(metadata)

    id = json.loads(metadata)["id"]

    sets = []

    page = 1

    resp = api.dataset_list(mine=True, search=name, page=page)

    while len(resp):
        sets += resp

        page += 1

        resp = api.dataset_list(mine=True, search=name, page=page)

    if id in [str(item) for item in sets]:
        api.dataset_create_version(dest, delete_old_versions=True, convert_to_csv=False, version_notes="new version", dir_mode="zip")
    else:
        api.dataset_create_new(dest, convert_to_csv=False, dir_mode="zip")

def main():
    parser = argparse.ArgumentParser(description='upload data to kaggle.')

    parser.add_argument('--dataset', type=str, help='dataset id', default=None)
    parser.add_argument('--experiments', type=str, help='experiments', default=None)

    args = parser.parse_args()

    print(args.dataset)
    print(args.experiments)

    if args.experiments:
        experiments = args.experiments.split(',')

        to_dataset(os.getcwd(), experiments, args.dataset)
    else:
        to_dataset(os.getcwd(), None, args.dataset, True)