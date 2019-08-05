import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import argparse
from musket_core.projects import Workspace
from musket_core import caches
from musket_core.tools import Launch,ProgressMonitor


def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--project', type=str, required=True,
                        help='folder to search for experiments')
    parser.add_argument('--name', type=str, default="",
                        help='name of the experiment')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--gpus_per_net', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--allow_resume', type=bool, default=True,
                        help='allow resuming of experiments')
    parser.add_argument('--force_recalc', type=bool, default=False,
                        help='force rebuild reports and predictions')
    parser.add_argument('--launch_tasks', type=bool, default=False,
                        help='launch associated tasks')
    parser.add_argument('--only_report', type=bool, default=False,
                        help='only generate reports')
    parser.add_argument('--cache', type=str, default="",
                        help='cache directory')
    args = parser.parse_args()
    if len(args.cache)>0:
        caches.CACHE_DIR = args.cache
    w=Workspace()
    project=w.project(args.project)

    experiments=project.experiments()

    if len(args.name)>0:
        mmm=args.name.split(",")
        res=[]
        for x in experiments:
            if x.name() in mmm:
                res.append(x)
        experiments=res
    else:
        experiments=[x for x in experiments if not x.isCompleted()]

    l=Launch(args.gpus_per_net,args.num_gpus,args.num_workers,[x.path for x in experiments],args.allow_resume,args.only_report,args.launch_tasks)
    l.perform(w,ProgressMonitor())
    exit(0)
    pass
if __name__ == '__main__':
    main()
