import os
import shutil
import requests

from musket_core import utils

def publish_local_project(host, project):
    destination = os.path.expanduser("~/.musket_core/zip_assembly")

    if os.path.exists(destination):
        shutil.rmtree(destination)

    utils.collect_project(project, destination)

    target_path = os.path.expanduser("~/.musket_core/zip_result")

    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    archive_path = os.path.join(target_path, "project")

    utils.archive(destination, archive_path)

    with open(archive_path + ".zip", 'rb') as zf:
        requests.post(host + "/zipfile", files={'file': ('file.zip', zf, 'application/zip')})

def clone_from_repo(host, git_url):
    requests.get(host + "/gitclone", params={"git_url": git_url})