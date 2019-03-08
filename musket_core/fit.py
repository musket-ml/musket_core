import argparse
import sys
import os
from keras import backend as K
from musket_core.utils import save_yaml,load_yaml
from musket_core import generic
import threading
import queue
import tensorflow as tf
import traceback

class Consumer(threading.Thread):
    def __init__(self, queue,num,respond,num_gpus):
        threading.Thread.__init__(self)
        self._queue = queue
        self.num=num
        self.num_gpus=num_gpus
        self.respond=respond

    def run(self):
         while True:
             exp = self._queue.get()
             try:

               if exp is None:
                   break
               else:
                   print("Performing:"+exp.path)
                   config = tf.ConfigProto()
                   sess = tf.Session(config=config)
                   with sess.as_default():
                       with tf.device("/gpu:"+str(self.num%2)):
                           exp.fit()
                           print(exp.metrics())
                           K.clear_session()
             finally:
                 self.respond.put(exp.path)



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
            print(self.path)
            print(exc_value)
            print(traceback.format_exc())
            print(exc_type)
            save_yaml(self.path+"/error.yaml",[str(exc_value),str(traceback.format_exc()),str(exc_type)])


def gather_work(path,e:[Experiment]):
    for d in os.listdir(path):
        fp=os.path.join(path, d)
        if os.path.isdir(fp):
            gather_work(fp,e)
        if d=="config.yaml":
            if not os.path.exists(path + "/summary.yaml") and not os.path.exists(path + "/started.yaml"):
                os.remove(path + "/summary.yaml")
            e.append(Experiment(path))

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
    q = queue.Queue()
    q1 = queue.Queue()
    workers = [Consumer(q, num,q1,args.num_gpus) for num in range(args.num_workers)]
    for w in workers:
        w.setDaemon(True)
        w.start()

    gather_work(expDir,todo)
    for exp in todo:
        exp.cleanup()
        q.put(exp)
    completed=[]
    while True:
        if len(completed)==len(todo):
            break
        completed.append(q1.get())
    exit(0)
    pass
if __name__ == '__main__':
    main()