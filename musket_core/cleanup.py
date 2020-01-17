import sys
import os
from builtins import bool
print(("Adding " + os.path.dirname(sys.path[0])).encode('ascii', 'replace'))
#sys.path.append(os.path.dirname(sys.path[0]))
sys.path[0] = os.path.dirname(sys.path[0])
print("sys.path:")
print(str(sys.path).replace('\\', '/').encode('ascii', 'replace'))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #,1,2"
import argparse
from musket_core.projects import Workspace
from musket_core import caches
from musket_core.tools import Cleanup,ProgressMonitor
import tensorflow as tf

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
    
    parser.add_argument('--cache', type=bool, default=True,
                        help='clean cache directory')
    
    args = parser.parse_args()    
    w=Workspace()
    project=w.project(args.project)

    experiments=project.experiments()

    if len(args.name)>0:
        mmm=args.name.split(",")
        res=[]
        for x in experiments:
            if x.name() in mmm:
                res.append(x)
        experiments = sorted(res, key = lambda x: mmm.index(x.name()))
    else:
        experiments=[x for x in experiments]

    l=Cleanup([x.path for x in experiments],args.cache)
    l.perform(w,ProgressMonitor())
    exit(0)
    pass
if __name__ == '__main__':
    main()