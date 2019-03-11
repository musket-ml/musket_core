from keras import backend as K
import threading
import queue
import tensorflow as tf

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


class Executor:

    def __init__(self,numWorkers,numGpus):
        self.q = queue.Queue()
        self.q1 = queue.Queue()
        self.workers = [Consumer(self.q, num, self.q1, numGpus) for num in range(numWorkers)]
        for w in self.workers:
            w.setDaemon(True)
            w.start()
        pass



    def execute(self,todo):
        for exp in todo:
            self.q.put(exp)
        completed = []
        while True:
            if len(completed) == len(todo):
                break
            completed.append(self.q1.get())

    def terminate(self):
        for w in self.workers:
            self.q1.put(None)

_executor=None
def get_executor(num_workers,numGpus):
    global _executor
    if _executor is not None:
        return _executor
    _executor=Executor(num_workers,numGpus)
    return _executor