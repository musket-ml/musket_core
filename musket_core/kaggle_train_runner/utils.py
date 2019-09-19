import os, subprocess, json, shutil, time

from musket_core.utils import templates_folder

import sys

try:
    import kaggle
except:
    print("can't import kaggle!")

api_holder = []

def resolve_template(path):
    return os.path.join(templates_folder(), path)

def run_kaggle_cmd(cmd):
    kaggle_path = templates_folder()

    command = os.path.join(kaggle_path, "cli.py")

    full_command = sys.executable + " " + command + " " + cmd;

    #full_command = "python --version"

    #print(full_command)

    return os.popen(full_command).read()

def run_cmd(cmd, retry_after = 10):
    #print(os.path.dirname(kaggle.__file__))

    #print(os.environ)

    #popen("pip install kaggle")

    try:
        #return popen(cmd).read()

        return run_kaggle_cmd(cmd[len("kaggle") + 1: ])
    except:
        print("exception while running: " + cmd)

        print("retry will be started after " + str(retry_after) + " seconds")

        time.sleep(retry_after)

        return run_cmd(cmd, retry_after)

def get_status(kernel_id):
    cmd = "kaggle kernels status " + kernel_id

    print("status_request: " + cmd)

    return run_cmd(cmd)

def get_status_api(kernel_id, retry_after = 10, after_run = False):
    splited = kernel_id.split("/")

    try:
        result = kaggle.api.kernel_status(splited[0], splited[1])

        if result["status"] == "error":
            return 'has status "error"'

        if result["status"] == "complete":
            return 'has status "complete"'

        if result["status"] == "running":
            return 'has status "running"'

    except Exception as e:
        if 'Not Found' in str(e):
            return "404 - Not Found"

        print("exception while running: get status")

        print("retry will be started after " + str(retry_after) + " seconds")

        time.sleep(retry_after)

        return get_status_api(kernel_id, retry_after)

def run_kernel(kernel_path):
    cmd = "kaggle kernels push -p " + kernel_path

    return run_cmd(cmd)

def run_kernel_api(kernel_path, retry_after = 10):
    try:
        kaggle.api.kernels_push(kernel_path)

        return "successfully runned: " + kernel_path
    except Exception as e:
        print(e)

        print("exception while running: run kernel")

        print("retry will be started after " + str(retry_after) + " seconds")

        time.sleep(retry_after)

        return run_kernel_api(kernel_path, retry_after)


def get_kernel_template():
    with open(resolve_template("kernel-metadata-template.json")) as template:
        result = json.load(template)

    return result

def get_datataset_template():
    with open(resolve_template("dataset-metadata-template.json")) as template:
        result = json.load(template)

    return result

def get_notebook_template(server, kernel_id, fold, time, kernel_ref):
    with open(resolve_template("notebook.ipynb")) as template:
        result = template.read().replace("server", server).replace("kernel_id", str(kernel_id))

        if fold >= 0:
            result = result.replace("folds_argument", str(fold))

        if time >= 0:
            result = result.replace("timer_argument", str(time))

        result = result.replace("kernel_ref", kernel_ref)

        return result

def read_project_meta(project_path):
    with open(os.path.join(project_path, ".metadata", "kaggle-project-metadata.json")) as meta:
        result = json.load(meta)

    return result

def is_complete(path):
    project_path = os.path.join(path, "project")

    experiment_path = os.path.join(project_path, "experiments")
    experiment_path = os.path.join(experiment_path, "experiment")

    in_progress_path = os.path.join(experiment_path, "inProgress.yaml")

    if os.path.exists(in_progress_path):
        return False

    return True

def download(id, dest):
    print("downloading: id")

    run_cmd("kaggle kernels output " + id + " -p " + dest)

def kernel_meta(kernel_path, kernel_id, username, server, title, dataset_sources, competition_sources, kernel_sources, fold, time):
    result = get_kernel_template()

    result["id"] = username + "/" + title
    result["title"] = title
    result["code_file"] = os.path.join(kernel_path, "notebook.ipynb")

    result["dataset_sources"] = dataset_sources
    result["competition_sources"] = competition_sources
    result["kernel_sources"] = kernel_sources + [username + "/" + title]

    ensure(os.path.dirname(kernel_path))
    ensure(kernel_path)

    with open(os.path.join(kernel_path, "kernel-metadata.json"), "w") as f:
        json.dump(result, f)

    with open(os.path.join(kernel_path, "notebook.ipynb"), "w") as f:
        f.write(get_notebook_template(server, kernel_id, fold, time, title))

    return result

def write_dataset_meta(username, title, path):
    result = get_datataset_template()

    result["id"] = username + "/" + title
    result["title"] = title

    with open(os.path.join(path, "dataset-metadata.json"), "w") as f:
        json.dump(result, f)

def ensure(path):
    if os.path.exists(path):
        return

    os.mkdir(path)

def archive(src, dst):
    shutil.make_archive(dst, 'zip', os.path.dirname(src), "project")

def log(path, bytes):
    with open(path, "a") as f:
        f.buffer.write(bytes)

def copy(src, dst):
    return shutil.copytree(src, dst, True)

def copy_file(src, dst):
    return shutil.copy(src, dst)

def remove(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def listdir(path):
    return [item for item in os.listdir(path) if not path == ".DS_Store"]