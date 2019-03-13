from musket_core.utils import save_yaml,load_yaml,ensure
import os
import numpy as np
import traceback
import random
import sys
from musket_core import generic


class Experiment:

    def __init__(self,path):
        self.path=path
        self._cfg=None

    def cleanup(self):
        if os.path.exists(self.path + "/predictions"):
            for pr in os.listdir(self.path + "/predictions"):
                os.remove(self.path + "/predictions/" + pr)
        if os.path.exists(self.path + "/summary.yaml"):
            os.remove(self.path + "/summary.yaml")

    def metrics(self):
        if os.path.exists(self.path+"/summary.yaml"):
            return load_yaml(self.path+"/summary.yaml")
        return {}

    def final_metric(self):
        m=self.config()
        return  m["final_metric"] if "final_metric" in m else None

    def hyperparameters(self):
        config = self.config()
        ps= config["hyperparameters"] if "hyperparameters" in config else None
        return ps

    def generateMetrics(self):
        cfg = self.parse_config()
        cfg.generateReports()

    def parse_config(self):
        if os.path.exists(self.path + "/config.yaml"):
            cfg = generic.parse(self.path + "/config.yaml")

        else:
            cfg = generic.parse(self.path + "/config_concrete.yaml")
        return cfg

    def result(self,forseRecalc=False):
        pi = self.apply(True)
        if forseRecalc:
            self.cleanup()
        m=self.metrics()
        if len(m)>0:
            return m
        if len(pi) > 1:
            vals = []
            for i in pi:
                if i.isCompleted() or True:
                    if forseRecalc:
                        i.cleanup()
                    i.generateMetrics()
                    m = i.metrics()
                    pm = i.config()["primary_metric"]
                    if "val_" in pm:
                        pm = pm[4:]
                    mv = m["allStages"][pm + "_holdout"]
                    if "aggregation_metric" in i.config():
                        mv = m["allStages"][i.config()["aggregation_metric"]]
                    vals.append(mv)
            m = np.mean(vals)
            save_yaml(self.path + "/summary.yaml",
                      {"mean": float(m), "max": float(np.max(vals)), "min": float(np.min(vals)),
                       "std": float(np.std(vals))})
            return float(m)
        else:
            m = self.metrics()
            return m
        pass

    def isCompleted(self):
        return os.path.exists(self.path+"/summary.yaml")

    def isStarted(self):
        return os.path.exists(self.path+"/started.yaml")

    def config(self):
        if self._cfg is not None:
            return self._cfg
        if os.path.exists(self.path + "/config.yaml"):
            self._cfg= load_yaml(self.path + "/config.yaml")

        else:
            self._cfg=load_yaml(self.path + "/config_concrete.yaml")
        return self._cfg

    def dumpTo(self,path,extra):
        c=self.config()
        for k in extra:
            c[k]=extra[k]
        return save_yaml(path + "/config_concrete.yaml",c)

    def fit(self):
        try:
            self.cleanup()
            save_yaml(self.path + "/started.yaml", True)
            cfg = self.parse_config()
            cfg.fit()
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(self.path)
            print(exc_value)
            print(traceback.format_exc())
            print(exc_type)
            save_yaml(self.path+"/error.yaml",[str(exc_value),str(traceback.format_exc()),str(exc_type)])


    def apply(self,all=False):
        if self.hyperparameters() is not None:
            return [self]
        m=self.config()
        if "num_seeds" in m:
            paths = []
            for i in range(m["num_seeds"]):
                i_ = self.path + "/" + str(i)
                ensure(i_)
                if not all:
                    if Experiment(i_).isCompleted():
                        continue

                s=random.randint(0,100000)
                self.dumpTo(i_, {"testSplitSeed":s})
                paths.append(Experiment(i_))
            return paths
        return [self]