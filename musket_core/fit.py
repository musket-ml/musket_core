import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import argparse
import sys
from musket_core.experiment import Experiment
from musket_core.structure_constants import isNewExperementDir
from musket_core.parralel import get_executor
from musket_core import hyper

def gather_work(path,e:[Experiment],name="",allow_resume=False):

    if isNewExperementDir(path):
        if len(name)>0 and name!=os.path.basename(path):
            return
        else:
            pi = Experiment(path,allow_resume).apply()
            for v in pi: e.append(v)
        return
    for d in os.listdir(path):
        fp=os.path.join(path, d)
        if os.path.isdir(fp):
            gather_work(fp,e,name,allow_resume)



def gather_stat(path,name="",forseRecalc=False):

    if isNewExperementDir(path):
        if len(name)>0 and name!=os.path.basename(path):
            return
        else:
            ex = Experiment(path)
            ex.result(forseRecalc)

    for d in os.listdir(path):
        fp=os.path.join(path, d)
        if os.path.isdir(fp):
            gather_stat(fp,name,forseRecalc)

def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--project', type=str, default=".",
                        help='folder to search for experiments')
    parser.add_argument('--name', type=str, default="",
                        help='name of the experiment')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--force_recalc', type=bool, default=False,
                        help='force rebuild reports and predictions')
    parser.add_argument('--allow_resume', type=bool, default=True,
                        help='allow resuming previously failed experiments')
    args = parser.parse_args()
    inf=args.project
    sys.path.insert(0, os.path.join(inf,"modules"))
    expDir=os.path.join(inf,"experiments")
    todo=[]
    e=get_executor(args.num_workers,args.num_gpus)

    gather_work(expDir,todo,args.name,args.allow_resume)

    hasHyper=[e for e in todo if e.hyperparameters() is not None]
    noHyper=[e for e in todo if e.hyperparameters() is  None]
    e.execute(noHyper)
    for h in hasHyper:
        hyper.optimize(h,e)
    gather_stat(expDir,args.name,args.force_recalc)
    exit(0)
    pass
if __name__ == '__main__':
    main()