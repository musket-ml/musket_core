from musket_core import experiment
from musket_core import parralel
import copy
from musket_core import templating
from musket_core.utils import *
from musket_core import tools
import numpy as np

from hyperopt import fmin, tpe, hp, Trials

num = 0
scores={}

def expand(value):
    if isinstance(value, list):
        res = {}

        res["range"] = v
        res["type"] = "float"

        return  res

    res = copy.deepcopy(value)

    if not "type" in res.keys():
        res["type"] = "float"

    return res

def create_parameter(name, value):
    val = expand(value)

    if "range" in val.keys():
        min = val["range"][0]
        max = val["range"][1]

        if val["type"] == "float":
            return hp.uniform(name, min, max)

        return hp.choice(name, list(range(min, max + 1)))

    enum = np.array(val["enum"])

    if val["type"] == "float":
        enum = enum.astype(np.float32)
    else:
        enum = enum.astype(np.int32)

    return hp.choice(name, [value.item() for value in enum])

def create_search_space(e:experiment.Experiment):
    space = {}

    for p in e.hyperparameters():
        space[p]=create_parameter(p,e.hyperparameters()[p])

    return space

def get_score(num_or_dict):
    if isinstance(num_or_dict, (float, int, np.number)):
        return num_or_dict

    return num_or_dict["mean"]

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
        num=max(scores.keys()) + 1

    header = None

    with open(os.path.join(e.path, "config.yaml")) as cfg_f:
        s = cfg_f.read()

        if s.split("\n")[0].startswith("#"):
            header = s.split("\n")[0]

    def doOptimize(parameters):
        save(e.path + "/hyperopt.trials", trials)

        global num,scores

        resolved=templating.resolveTemplates(copy.deepcopy(dcopy), parameters)

        num_ = e.path + "/trial" + str(num)

        ensure(num_)

        save_yaml(num_ + "/config.yaml", resolved, header)

        experiment_experiment = experiment.Experiment(num_)

        tasks=experiment_experiment.fit(reporter)

        if reporter.isCanceled():
            raise OperationCanceled()

        ex.execute(tasks)

        num = num + 1

        ex_result = experiment_experiment.result(False, True)

        score = get_score(ex_result["value"])

        scores[num] = score

        save_yaml(e.path + "/hyperopt.scores", scores)

        if reporter.isCanceled():
            raise OperationCanceled()

        return score if ex_result["mode"] == "min" else -score

    try:
        best = fmin(doOptimize, space=space, trials=trials, algo=tpe.suggest, max_evals=max_evals)
    except OperationCanceled:
        return

    save_yaml(e.path+"/best.params.yaml",best)

    best["mean"]=float(np.mean(list(scores.values())))
    best["max"] =float(np.max(list(scores.values())))
    best["min"] = float(np.min(list(scores.values())))

    save_yaml(e.path+"/summary.yaml",best)

    return best