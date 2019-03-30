from threading import local,Thread
from tqdm import tqdm
import os
import numpy as np
from musket_core.utils import save,load

_context=local()

class Visualizer:

    def __init__(self,func,path,dataset):
        self.func=func
        self._cache={}
        self.dataset=dataset
        self.path=path
        if os.path.exists(self._indexPath()):
            self._cache=load(self._indexPath())
        pass

    def __getitem__(self, item):
        _context.path=self.path
        if item in self._cache:
            return self._cache[item]
        val=self.func(self.dataset[item])
        self._cache[item]=val
        return val

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

def require_original(func):
    func.visualizer=True
    def rs(x):
        return func(x.rootItem())
    rs.__name__=func.__name__
    rs.visualizer=True
    return rs

def dataset_analizer(func):
    func.analizer=True
    return func

def context():
    return _context