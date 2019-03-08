import argparse
import sys
import os
from keras import backend as K
from musket_core.utils import save_yaml,load_yaml

from musket_core import generic
import time
import threading
import queue
import tensorflow


class Consumer(threading.Thread):
    def __init__(self, queue,num):
        threading.Thread.__init__(self)
        self._queue = queue
        self.num=num

    def run(self):
         while True:
           exp = self._queue.get()
           if exp is None:
               break
           else:
               print("Performing:"+exp.path)
               with K.tf.device("/gpu:0"):
                   with K.tf.Session().as_default():
                       exp.fit()
                       print(exp.metrics())
                       K.clear_session()

queue = queue.Queue()
workers=[Consumer(queue,num) for num in range(4)]
for w in workers:
    w.setDaemon(True)
    w.start()

class Experiment:

    def __init__(self,path):
        self.path=path

    def cleanup(self):
        if os.path.exists(self.path + "/predictions"):
            for pr in os.listdir(self.path + "/predictions"):
                os.remove(self.path + "/predictions/" + pr)

    def metrics(self):
        if os.path.exists(self.path+"/summary.yaml"):
            return load_yaml(self.path+"/summary.yaml")
        return {}

    def isCompleted(self):
        return os.path.exists(self.path+"/summary.yaml")

    def isStarted(self):
        return os.path.exists(self.path+"/started.yaml")

    def fit(self):
        try:
            save_yaml(self.path + "/started.yaml", True)
            cfg = generic.parse(self.path + "/config.yaml")
            cfg.fit()
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)
            print(exc_traceback)
            print(exc_type)
            save_yaml(self.path+"/error.yaml",[exc_value,exc_traceback,exc_type])


def gather_work(path,e:[Experiment]):
    for d in os.listdir(path):
        fp=os.path.join(path, d)
        if os.path.isdir(fp):
            gather_work(fp,e)
        if d=="config.yaml":
            if not os.path.exists(path + "/summary.yaml") and not os.path.exists(path + "/started.yaml"):
                e.append(Experiment(path))

def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--project', type=str, default=".",
                        help='folder to search for experiments')
    args = parser.parse_args()
    inf=args.project
    sys.path.insert(0, os.path.join(inf,"modules"))
    expDir=os.path.join(inf,"experiments")
    todo=[]
    gather_work(expDir,todo)
    for exp in todo:
        exp.cleanup()
        queue.put(exp)
    queue.join()


    pass
if __name__ == '__main__':
    main()