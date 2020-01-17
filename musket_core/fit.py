import sys
import os
from musket_core.fit_callbacks import after_fit_callbacks
print("Adding " + os.path.dirname(sys.path[0]))
#sys.path.append(os.path.dirname(sys.path[0]))
sys.path[0] = os.path.dirname(sys.path[0])
print("sys.path:")
print(sys.path)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #,1,2"
import argparse
from musket_core.projects import Workspace
from musket_core import caches, deps_download, fit_callbacks
from musket_core.tools import Launch,ProgressMonitor
import tensorflow as tf
from multiprocessing import Process

try:
    #do not remove
    import torch
except:
    pass

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
    parser.add_argument('--allow_resume', type=bool, default=False,
                        help='allow resuming of experiments')
    parser.add_argument('--force_recalc', type=bool, default=False,
                        help='force rebuild reports and predictions')
    parser.add_argument('--launch_tasks', type=bool, default=False,
                        help='launch associated tasks')
    parser.add_argument('--only_report', type=bool, default=False,
                        help='only generate reports')
    parser.add_argument('--cache', type=str, default="",
                        help='cache directory')
    parser.add_argument('--folds', type=str, default=None,
                        help='folds to execute')
    parser.add_argument('--time', type=int, default=-1,
                        help='time to work')
    parser.add_argument('--one_process', type=bool, default=False,
                        help='Start all experiments in one process. Good to use for several small experiments')
    parser.add_argument('-d','--download_deps', action='store_true',
                    help='download dependencies (e.g. dataset) prior to fitting')

    args = parser.parse_args()
    if len(args.cache)>0:
        caches.CACHE_DIR = args.cache
    workspace=Workspace()
    project=workspace.project(args.project)

    experiments=project.experiments()

    if len(args.name)>0:
        mmm=args.name.split(",")
        res=[]
        for x in experiments:
            if x.name() in mmm:
                res.append(x)
        experiments = sorted(res, key = lambda x: mmm.index(x.name()))
    else:
        experiments=[x for x in experiments if not x.isCompleted()]

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    folds = args.folds

    if folds:
        folds = [int(item) for item in folds.split(',')]
        
    if args.download_deps:
        deps_download.main(sys.argv)
    if len(experiments) == 0:
        print("No experiments or all experiments already finished, nothing to launch")    
    elif args.one_process or len(experiments) == 1:
        perform_experiments(workspace, args.gpus_per_net,args.num_gpus,args.num_workers,[x.path for x in experiments],args.allow_resume,args.only_report,args.launch_tasks, folds, args.time)
        print("===== All experiments finished =====")
    else:
        for x in experiments:
            p = Process(target=perform_experiments, args=(workspace,args.gpus_per_net,args.num_gpus,args.num_workers,[x.path],args.allow_resume,args.only_report,args.launch_tasks, folds, args.time))           
            p.start()
            p.join()               
        print("===== All experiments finished =====")
    
    callbacks = fit_callbacks.get_after_fit_callbacks()
    if (len(callbacks) > 0):
        print("Running {} after-fit callbacks".format(len(callbacks)))
    for func in callbacks:
        print("Callback {}".format(func.__name__))        
        func()
    
    exit(0)
    pass

def perform_experiments(workspace, gpus_per_net,num_gpus,num_workers, experiment_paths,allow_resume,only_report,launch_tasks, folds, time):
    l=Launch(gpus_per_net,num_gpus,num_workers,experiment_paths,allow_resume,only_report,launch_tasks, folds, time)
    l.perform(workspace,ProgressMonitor())


if __name__ == '__main__':
    main()