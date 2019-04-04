from threading import local,Thread
from musket_core.parralel import Error
from tqdm import tqdm

import os
import numpy as np
from musket_core.utils import save,load
from musket_core import parralel

_context=local()

class VFunc:

    def __init__(self,dataset,item,func,args,path):
        self.dataset=dataset
        self.item=item
        self.func=func
        self.path=path
        self.args=args
        pass

    def __call__(self, *args, **kwargs):
        _context.path = self.path
        if self.args is not None:
            val = self.func(self.dataset[self.item], **self.args)
        else:
            val = self.func(self.dataset[self.item])
        return val

class Visualizer:

    def __init__(self,func,path,dataset):
        self.func=func
        self._cache={}
        self.dataset=dataset
        self.path=path
        self.args=None
        if os.path.exists(self._indexPath()):
            self._cache=load(self._indexPath())
        pass

    def __getitem__(self, item):
        try:
            _context.path=self.path

            if item in self._cache:
                return self._cache[item]

            t = parralel.Task(VFunc(self.dataset, item, self.func, self.args,self.path),requiresSession=False);
            parralel.get_visualizer_executor().execute([t])


            self._cache[item]=t.result
            return t.result
        except:
            Error()
            return None
    def __len__(self):
        return len(self.dataset)

    def all(self,workers=None):
        if workers is not None:
            ws=np.array_split(np.arange(0,len(self)),workers)
            threads=[]
            for w in ws:
                def process():
                    for i in tqdm(range(len(w))):
                        v = self[w[i]]
                t1=Thread(target=process)
                t1.setDaemon(True)
                t1.start()
                threads.append(t1)
            for t in threads:
                t.join()

        else:
            for i in tqdm(range(len(self))):
                v = self[i]
        save(self._indexPath(), self._cache)

    def _indexPath(self):
        return os.path.join(self.path, "index.data")


def dataset_visualizer(func):
    func.visualizer=True
    return func

def visualize_as_image(func):
    func.viewer="image"
    return func

def visualize_as_text(func):
    func.viewer="text"
    return func

def visualize_as_html(func):
    func.viewer="html"
    return func

def require_original(func):
    func.visualizer=True
    def rs(x,**kwargs):
        return func(x.rootItem(),**kwargs)
    rs.__name__=func.__name__
    rs.visualizer=True
    rs.original=func
    if hasattr(func,"viewer"):
        rs.viewer=getattr(func,"viewer")
    return rs

def prediction_analizer(func):
    func.analizer=True
    return func

def dataset_analizer(func):
    func.data_analizer=True
    return func

def context():
    return _context