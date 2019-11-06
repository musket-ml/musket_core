import sys
import os

from musket_core import fit

def run():
    if len(sys.argv) < 2:
        raise("'run_experiment' or 'run_project' should be specified")
    
    task = sys.argv[1]
    
    if task == "run_experiment":
        run_experiment()
    elif task == "run_project":
        run_project()
    else:
        raise("'run_experiment' or 'run_project' should be specified")

def run_experiment():
    cwd = os.getcwd()

    experiment_name = os.path.basename(cwd)

    argv = []

    argv.append(__file__)

    argv.append("--project")
    argv.append(os.path.abspath(os.path.join(cwd, "../../")))

    argv.append("--name")
    argv.append(experiment_name)

    argv.append("--allow_resume")
    argv.append("true")

    sys.argv = argv

    fit.main()

def run_project():
    cwd = os.getcwd()

    argv = []

    argv.append(__file__)

    argv.append("--project")
    argv.append(cwd)

    argv.append("--allow_resume")
    argv.append("true")

    sys.argv = argv

    fit.main()