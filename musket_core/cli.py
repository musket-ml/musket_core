import sys
import os

FIT = "fit"
ANALYZE = "analyze"
DOWNLOAD_DEPS = "deps_download"

def is_experiment(root):
    config_path = os.path.join(root, "config.yaml")

    return os.path.exists(config_path)

def project_path():
    cwd = os.getcwd()

    if is_experiment(cwd):
        return os.path.abspath(os.path.join(cwd, "../../"))

    return cwd

def convert_args(root, task_name):
    args = sys.argv

    new_args = [os.path.join(os.path.dirname(__file__), task_name + ".py")] + list(args[2:])

    if task_name == FIT and not "--project" in new_args:
        new_args.append("--project")
        new_args.append(project_path())

    if task_name == FIT and not "--name" in new_args:
        if is_experiment(root):
            new_args.append("--name")
            new_args.append(experiment_name())

    elif task_name == ANALYZE and not "--inputFolder" in new_args:
        new_args.append("--inputFolder")
        new_args.append(project_path())

    elif task_name == DOWNLOAD_DEPS:
        new_args.insert(1, project_path())

    sys.argv = new_args

def main():
    if len(sys.argv) < 2:
        print("no task specified, command should be one of:")
        print("musket " + FIT)
        print("musket " + ANALYZE)
        print("musket " + DOWNLOAD_DEPS)

        return

    task_name = sys.argv[1]

    if task_name not in [FIT, ANALYZE, DOWNLOAD_DEPS]:
        print("unknown task: " + task_name + ", command should be one of:")
        print("musket " + FIT)
        print("musket " + ANALYZE)
        print("musket " + DOWNLOAD_DEPS)

        return

    from musket_core import fit, analize, deps_download

    convert_args(os.getcwd(), task_name)

    if task_name == FIT:
        fit.main()
    elif task_name == ANALYZE:
        analize.main()
    elif task_name == DOWNLOAD_DEPS:
        deps_download.main(sys.argv)

def experiment_name():
    cwd = os.getcwd()
    return os.path.basename(cwd)