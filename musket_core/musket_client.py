import os
import io
import sys

import shutil
import requests

from musket_core import utils

import time

import argparse

def publish_local_project(host, project, **kwargs):
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
        response = requests.post(host + "/zipfile", files={'file': ('file.zip', zf, 'application/zip')})

    if response.status_code == 200:
        project_fit(host, lambda data: print(data), project=response.text, **kwargs)

def project_fit(host, on_report, **kwargs):
    url = host + "/project_fit"

    response = requests.get(url, kwargs)

    if response.status_code == 200:
        task_id = response.text

        if not task_id == "unknown":
            get_report(host, task_id, on_report)

def get_report(host, task_id, on_report, is_last = False):
    url = host + ("/last_report" if is_last else "/report")

    from_line = 0

    reporting = True

    while reporting:
        try:
            response = requests.get(url, params={"task_id": task_id, "from_line": from_line})

            if response.status_code == 200:
                for line in io.TextIOWrapper(io.BytesIO(response.content), encoding="utf-8"):
                    if line == "empty_string":
                        break

                    if line == "report_not_awailable_yet":
                        print("awaiting report...")

                        break

                    if line == "report_task_scheduled":
                        print("task sheduled but not started yet. request later:")
                        print(host + "/last_report?task_id=" + task_id)
                        print("or use terminal cmd:")
                        print("musket server_task_report " + task_id)

                        reporting = False

                        break

                    on_report(line)

                    from_line += 1

                    if is_last:
                        reporting = False

                    if line == "report_end":
                        reporting = False

            else:
                reporting = False

            time.sleep(5)
        except Exception as e:
            print(e)

            reporting = False

            on_report("something wrong. abort.")

def clone_from_repo(host, git_url, **kwargs):
    response = requests.get(host + "/gitclone", params={"git_url": git_url})

    if response.status_code == 200:
        project_fit(host, lambda data: print(data), project=response.text, **kwargs)

def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--host', type=str, required=True, help='server URL')
    parser.add_argument('--git', type=str, default=None, help='project on git URL')
    parser.add_argument('--report', type=str, default=None, help='task_id')

    parser.add_argument('--name', type=str, default="", help='name of the experiment')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpus_per_net', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--allow_resume', type=bool, default=False, help='allow resuming of experiments')
    parser.add_argument('--force_recalc', type=bool, default=False, help='force rebuild reports and predictions')
    parser.add_argument('--launch_tasks', type=bool, default=False, help='launch associated tasks')
    parser.add_argument('--only_report', type=bool, default=False, help='only generate reports')
    parser.add_argument('--cache', type=str, default="", help='cache directory')
    parser.add_argument('--folds', type=str, default=None, help='folds to execute')

    args = parser.parse_args()

    client_args = dict(args.__dict__)

    for key in ["git", "host", "report"]:
        client_args.pop(key, None)

    if args.git:
        clone_from_repo(args.host, args.git, **client_args)
    elif args.report:
        get_report(args.host, args.report, lambda data: print(data), True)
    else:
        publish_local_project(args.host, os.getcwd(), **client_args)

    print(args)

# publish_local_project("http://127.0.0.1:9393", "/Users/dreamflyer/Desktop/exp")

