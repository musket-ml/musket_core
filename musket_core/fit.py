import argparse
import sys
import os
from keras import backend as K

from musket_core import generic

def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--project', type=str, default=".",
                        help='folder to search for experiments')

    args = parser.parse_args()
    inf=args.project
    sys.path.insert(0, os.path.join(inf,"modules"))
    expDir=os.path.join(inf,"experiments")
    todo=os.listdir(expDir)
    for exp in todo:
        e=os.path.join(expDir,exp)
        if not os.path.exists(e+"/summary.yaml"):
            if os.path.exists(e+"/config.yaml"):
                cfg=generic.parse(e+"/config.yaml")
                print("Training:"+e)
                cfg.fit()
                K.clear_session()
    pass

if __name__ == '__main__':
    main()