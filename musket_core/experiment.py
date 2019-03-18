from musket_core.utils import save_yaml,load_yaml,ensure,delete_file,load_string,save_string
import os
import numpy as np
import traceback
import random
import sys
from musket_core import generic
from musket_core.structure_constants import constructPredictionsDirPath, constructSummaryYamlPath, constructInProgressYamlPath, constructConfigYamlPath, constructErrorYamlPath, constructConfigYamlConcretePath

class Experiment:

    def __init__(self,path,allowResume = False,project=None):
        self.path=path
        self._cfg=None
        self.project=project
        self.allowResume = allowResume

    def cleanup(self):
        if os.path.exists(self.getPredictionsDirPath()):
            for pr in os.listdir(self.getPredictionsDirPath()):
                os.remove(f"{self.getPredictionsDirPath()}/{pr}")
        if os.path.exists(self.getSummaryYamlPath()):
            os.remove(self.getSummaryYamlPath())

    def metrics(self):
        if os.path.exists(self.getSummaryYamlPath()):
            return load_yaml(self.getSummaryYamlPath())
        return {}


    def name(self):
        if self.project is not None:
            return self.path[len(self.project.experimentsPath())+1:].replace("\\","/")
        return os.path.basename(self.path)

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
        if os.path.exists(self.getConfigYamlPath()):
            cfg = generic.parse(self.getConfigYamlPath())

        else:
            cfg = generic.parse(self.getConfigYamlConcretePath())
        if self.allowResume:
            cfg.setAllowResume(self.allowResume)
        return cfg

    def log_path(self,fold,stage):
        return os.path.join(self.path,"metrics","metrics-"+str(fold)+"."+str(stage)+".csv")

    def report(self):
        cf=self.parse_config()
        path = os.path.join(os.path.dirname(__file__),"templates","logs.html")
        template=load_string(path)
        eex=self.apply()
        for e in eex:
            for i in range(cf.folds_count):
                for j in range(len(cf.stages)):
                    if os.path.exists(e.log_path(i,j)):
                        m=load_string(e.log_path(i,j))
                        ensure(os.path.join(e.path,"reports"))
                        rp=os.path.join(e.path, "reports", "report-" + str(i) + "." + str(j) + ".html")
                        save_string(rp,template.replace("${metrics}",m))



    def result(self,forseRecalc=False):
        pi = self.apply(True)
        if forseRecalc:
            self.cleanup()
        m=self.metrics()
        if m is None:
            return None
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
            save_yaml(self.getSummaryYamlPath(),
                      {"mean": float(m), "max": float(np.max(vals)), "min": float(np.min(vals)),
                       "std": float(np.std(vals))})
            return float(m)
        else:
            m = self.metrics()
            return m
        pass

    def isCompleted(self):
        return os.path.exists(self.getSummaryYamlPath())

    def isInProgress(self):
        return os.path.exists(self.getInProgressYamlPath())

    def setInProgress(self, val:bool):
        if val and not self.isInProgress():
            save_yaml(self.getInProgressYamlPath(), True)
        elif not val and self.isInProgress():
            delete_file(self.getInProgressYamlPath())


    def config(self):
        if self._cfg is not None:
            return self._cfg
        if os.path.exists(self.getConfigYamlPath()):
            self._cfg= load_yaml(self.getConfigYamlPath())

        else:
            self._cfg=load_yaml(self.getConfigYamlConcretePath())
        return self._cfg

    def dumpTo(self,path,extra):
        c=self.config()
        for k in extra:
            c[k]=extra[k]
        return save_yaml(constructConfigYamlConcretePath(path),c)

    def fit(self):
        try:
            self.cleanup()
            self.setInProgress(True)
            cfg = self.parse_config()
            cfg.fit()
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(self.path)
            print(exc_value)
            print(traceback.format_exc())
            print(exc_type)
            save_yaml(self.getErrorYamlPath(), [str(exc_value),str(traceback.format_exc()),str(exc_type)])
        finally:
            self.setInProgress(False)


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
                paths.append(Experiment(i_,self.allowResume))
            return paths
        return [self]

    def getSummaryYamlPath(self):
        return constructSummaryYamlPath(self.path)

    def getInProgressYamlPath(self):
        return constructInProgressYamlPath(self.path)

    def getConfigYamlPath(self):
        return constructConfigYamlPath(self.path)

    def getConfigYamlConcretePath(self):
        return constructConfigYamlConcretePath(self.path)

    def getErrorYamlPath(self):
        return constructErrorYamlPath(self.path)

    def getPredictionsDirPath(self):
        return constructPredictionsDirPath(self.path)
