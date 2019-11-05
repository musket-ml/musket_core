import sys
import os

from musket_core import fit

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