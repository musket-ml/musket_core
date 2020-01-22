import sys
from musket_core.project_paths import *

FIT = "fit"
ANALYZE = "analyze"
DOWNLOAD_DEPS = "deps_download"
CLIENT = "client"
CLEAN = "clean"
KAGGLE_UPLOAD = "kaggle_upload"

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
        print("musket " + CLIENT)
        print("musket " + KAGGLE_UPLOAD)

        return

    task_name = sys.argv[1]

    if task_name not in [FIT, ANALYZE, DOWNLOAD_DEPS, CLIENT, KAGGLE_UPLOAD]:
        print("unknown task: " + task_name + ", command should be one of:")
        print("musket " + FIT)
        print("musket " + ANALYZE)
        print("musket " + DOWNLOAD_DEPS)
        print("musket " + CLIENT)
        print("musket " + KAGGLE_UPLOAD)

        return

    convert_args(os.getcwd(), task_name)

    if task_name == CLEAN:
        from musket_core import cleanup

        cleanup.main()

    if task_name == FIT:
        from musket_core import fit

        fit.main()

    elif task_name == ANALYZE:
        from musket_core import analize

        analize.main()

    elif task_name == DOWNLOAD_DEPS:
        from musket_core import deps_download

        deps_download.main(sys.argv)

    elif task_name == CLIENT:
        from musket_core import musket_client

        musket_client.main()

    elif task_name == KAGGLE_UPLOAD:
        from musket_core import publish_as_dataset

        publish_as_dataset.main()

def experiment_name():
    cwd = os.getcwd()

    return os.path.basename(cwd)

if __name__ == '__main__':
    main()