from musket_core import experiment
import pickle
import time
from musket_core import parralel
import copy
from musket_core import templating
from musket_core.utils import *

from hyperopt import fmin, tpe, hp, STATUS_OK,Trials

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
def optimize(e:experiment.Experiment,ex:parralel.Executor):
    space=create_search_space(e)
    dcopy=e.config().copy()
    max_evals=e.config()["max_evals"]
    trials = Trials()
    del dcopy["hyperparameters"]
    del dcopy["max_evals"]
    if os.path.exists(e.path+"/hyperopt.trials"):
        trials=load(e.path+"/hyperopt.trials")
    def doOptimize(parameters):
        save(e.path+"/hyperopt.trials",trials)
        global num
        for p in parameters:
            parameters[p]=round(parameters[p])
        resolved=templating.resolveTemplates(copy.deepcopy(dcopy),parameters)

        num_ = e.path + "/trial" + str(num)
        ensure(num_)
        save_yaml(num_ + "/config.yaml", resolved)

        experiment_experiment = experiment.Experiment(num_)
        smallExps= experiment_experiment.apply()

        ex.execute(smallExps)
        num=num+1
        print(num)
        return -experiment_experiment.result()
    best = fmin(doOptimize,
        space=space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=max_evals)



