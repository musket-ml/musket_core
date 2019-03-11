import argparse
import sys
from musket_core.experiment import Experiment
from musket_core.parralel import get_executor
import os
from musket_core import hyper

def gather_work(path,e:[Experiment]):
    for d in os.listdir(path):
        fp=os.path.join(path, d)
        if os.path.isdir(fp):
            gather_work(fp,e)
        if d=="config.yaml":
            if not os.path.exists(path + "/summary.yaml") and not os.path.exists(path + "/started.yaml"):
                pi=Experiment(path).apply()
                for v in pi: e.append(v)


def gather_stat(path):
    for d in os.listdir(path):
        fp=os.path.join(path, d)
        if os.path.isdir(fp):
            gather_stat(fp)
        if d=="config.yaml":
            if not os.path.exists(path + "/summary.yaml") and not os.path.exists(path + "/started.yaml") or True:
                ex=Experiment(path)
                ex.result()

def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--project', type=str, default=".",
                        help='folder to search for experiments')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers')
    args = parser.parse_args()
    inf=args.project
    sys.path.insert(0, os.path.join(inf,"modules"))
    expDir=os.path.join(inf,"experiments")
    todo=[]
    e=get_executor(args.num_workers,args.num_gpus)

    gather_work(expDir,todo)
    hasHyper=[e for e in todo if e.hyperparameters() is not None]
    noHyper=[e for e in todo if e.hyperparameters() is  None]
    e.execute(noHyper)
    for h in hasHyper:
        hyper.optimize(h,e)
    gather_stat(expDir)
    exit(0)
    pass
if __name__ == '__main__':
    main()