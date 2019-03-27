import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
import argparse
from musket_core.projects import Workspace
from musket_core.tools import Introspect,ProgressMonitor


def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--project', type=str, required=True,
                        help='folder to search for experiments')
    parser.add_argument('--out', type=str, default="meta.yaml",
                        help='path to store project meta')

    args = parser.parse_args()
    w=Workspace()
    project=w.project(args.project)

    l=Introspect(args.project,args.out)
    l.perform(w,ProgressMonitor())
    exit(0)
    pass
if __name__ == '__main__':
    main()