from musket_core import experiment
from musket_core import parralel
import copy
from musket_core import templating
from musket_core.utils import *
from musket_core import tools
import numpy as np

from hyperopt import fmin, tpe, hp, Trials

def create_parameter(name,v):
    if isinstance(v,list):
        return hp.uniform(name,v[0],v[1])
    return None

def create_search_space(e:experiment.Experiment):
    space={}
    for p in e.hyperparameters():
        space[p]=create_parameter(p,e.hyperparameters()[p])
    return space
num = 0
scores={}


class OperationCanceled(BaseException):
    def __init__(self):
        pass

def optimize(e:experiment.Experiment,ex:parralel.Executor,reporter):
    global num, scores
    num=0
    scores={}
    space=create_search_space(e)
    dcopy=e.config().copy()
    max_evals=e.config()["max_evals"]
    trials = Trials()
    del dcopy["hyperparameters"]
    del dcopy["max_evals"]
    if os.path.exists(e.path+"/hyperopt.trials") and e.allowResume:
        trials=load(e.path+"/hyperopt.trials")
    if os.path.exists(e.path+"/hyperopt.scores") and e.allowResume:
        scores=load_yaml(e.path+"/hyperopt.scores")
        num=max(scores.keys())+1
    def doOptimize(parameters):

        save(e.path+"/hyperopt.trials",trials)
        global num,scores
        for p in parameters:
            parameters[p]=round(parameters[p])
        resolved=templating.resolveTemplates(copy.deepcopy(dcopy),parameters)

        num_ = e.path + "/trial" + str(num)

        ensure(num_)
        save_yaml(num_ + "/config.yaml", resolved)

        experiment_experiment = experiment.Experiment(num_)

        tasks=experiment_experiment.fit(reporter)
        if reporter.isCanceled():
            raise OperationCanceled()
        ex.execute(tasks)
        num=num+1
        print(num)
        score=experiment_experiment.result()
        scores[num]=score
        save_yaml(e.path + "/hyperopt.scores",scores)
        if reporter.isCanceled():
            raise OperationCanceled()
        #save_yaml("trials.log",scores)
        return -score
    try:

        best = fmin(doOptimize,
            space=space,
            trials=trials,
            algo=tpe.suggest,
            max_evals=max_evals)
    except OperationCanceled:
        return
    save_yaml(e.path+"/best.params.yaml",best)
    best["mean"]=float(np.mean(list(scores.values())))
    best["max"] =float(np.max(list(scores.values())))
    best["min"] = float(np.min(list(scores.values())))
    save_yaml(e.path+"/summary.yaml",best)
    return best