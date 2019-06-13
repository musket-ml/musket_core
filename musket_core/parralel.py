from keras import backend as K
import threading
import queue
import tensorflow as tf
import typing
import sys
import traceback

class Error:
    def __init__(self):
        self.exc_type, self.exc_value, self.exc_traceback = sys.exc_info()
        self.exc_traceback=traceback.format_exc()
        print(self.exc_traceback)
        print(self.exc_value,self.exc_type)
    def log(self):
        return { "type":str(self.exc_type),"value":str(self.exc_value),"trace":self.exc_traceback}


class Task:

    def __init__(self, func, requiresSession=True, runOnErrors=False, name="", needs_tasks=False):
        self.func=func
        self.name=name
        self.deps:typing.List[Task]=[]
        self.completed=False
        self.aborted=False
        self.run_on_errors=runOnErrors
        self.requiresSession=requiresSession
        self.needsTasks=needs_tasks
        self.result=None
        self.exception=None

    def __str__(self):
        if len(self.name)>0:
            return self.name
        return str(self.func)

    def has_not_complete_deps(self):
        for i in self.deps:
            if not i.completed and not i.aborted:
                return True
        return False

    def can_run(self):
        if self.run_on_errors:
            return True
        for i in self.deps:
            if not i.completed or i.aborted:
                return False
        return True

    def all_errors(self):
        rs=[]
        for i in self.deps:
            rs=rs+i.all_errors()
        if self.exception is not None:
            rs.append(self.exception)
        return list(set(rs))

    def run(self):
        try:
            if self.needsTasks:
                self.result=self.func(self)
            else: self.result=self.func()
            self.completed=True
        except:
            self.aborted=True
            self.exception=Error()



class Worker(threading.Thread):
    def __init__(self, queue:queue.Queue,num,respond,num_gpus):
        threading.Thread.__init__(self)
        self._queue = queue
        self.num=num
        self.num_gpus=num_gpus
        self.respond=respond

    def run(self):
        gpus = self.num % self.num_gpus
        while True:
            exp: Task = self._queue.get()
            if exp is None:
                break
            if exp.has_not_complete_deps():
                self._queue.put_nowait(exp)
                continue
            if not exp.can_run():
                exp.aborted = True
                self.respond.put(exp)
                continue
            if exp.requiresSession:
                if _executor.num_workers>1:
                    with self.create_session(gpus).as_default():
                        with tf.device("/gpu:" + str(gpus)):
                                try:
                                    exp.run()
                                finally:
                                     K.clear_session()
                                     self.respond.put(exp)
                else:
                    try:
                        with self.create_session(gpus).as_default():
                            exp.run()
                    finally:
                        K.clear_session()
                        self.respond.put(exp)
            else:
                exp.run()
                self.respond.put(exp)

    def create_session(self,gpus):

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        #K.set_session(sess)
        return sess



class Executor:

    def __init__(self,numWorkers,numGpus):
        self.q = queue.Queue()
        self.num_workers=numWorkers
        self.numGpus=numGpus
        self.q1 = queue.Queue()
        self.workers = [Worker(self.q, num, self.q1, numGpus) for num in range(numWorkers)]
        for w in self.workers:
            w.setDaemon(True)
            w.start()
        pass

    def execute(self,todo:typing.Collection[Task]):
        for exp in todo:
                self.q.put(exp)
        completed = []
        while True:
            if len(completed) == len(todo):
                break
            completed.append(self.q1.get())
        return completed

    def terminate(self):
        for w in self.workers:
            self.q1.put(None)

_executor=None
def get_executor(num_workers,numGpus):
    global _executor
    if _executor is not None:
        if _executor.num_workers!=num_workers or _executor.numGpus!=numGpus:
                _executor.terminate()
                _executor = Executor(num_workers, numGpus)
        return _executor
    _executor=Executor(num_workers,numGpus)
    return _executor


def schedule(todo:typing.Collection[Task]):
    global _executor
    if _executor is None:
        _executor=get_executor(1,1)
    _executor.execute(todo)

v_executor=None
def get_visualizer_executor()->Executor:
    global v_executor
    if v_executor is not None:
        return v_executor
    v_executor=Executor(1,1)
    return v_executor