from musket_core import parralel
from musket_core import hyper
from musket_core import model
from musket_core import datasets
from musket_core.generic_config import GenericTaskConfig
from musket_core.utils import save_yaml
from musket_core import projects
from musket_core.experiment import Experiment
import numpy as np
import tempfile
from tqdm import tqdm
class ProgressMonitor:

    def isCanceled(self)->bool:
        return False

    def task(self,message:str,totalWork:int):
        return False

    def worked(self,numWorked)->bool:
        return False

    def error(self,message:str):
        pass

    def stdout(self,message):
        pass

    def stderr(self,message):
        pass

    def done(self):
        return False

import yaml

from musket_core import java_server

class TaskLaunch(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.dside.ui.TaskConfiguration'

    def __init__(self, gpusPerNet, numGpus, numWorkers, experiments, allowResume=False, onlyReports: bool = False,
                 tasks: [str] = ("all",),tasksArgs=None):
        self.gpusPerNet = gpusPerNet
        self.numGpus = numGpus
        self.numWorkers = numWorkers
        self.experiments = experiments
        self.allowResume = allowResume
        self.onlyReports = onlyReports
        self.tasks = tasks
        self.taskArgs=tasksArgs
        if isinstance(self.tasks,str):
            self.tasks=self.tasks.split(",")
        pass

    def perform(self,server,reporter:ProgressMonitor):
        if isinstance(self.tasks,str):
            self.tasks=self.tasks.split(",")
        workPerProject={}
        for e in self.experiments:
            inde=e.index("experiments")
            pp=e[0:inde]
            localPath=e
            if pp in workPerProject:
                workPerProject[pp].append(localPath)
            else:
                workPerProject[pp]=[localPath]
        executor = parralel.get_executor(self.numWorkers, self.numGpus)
        rs=[]
        for projectPath in workPerProject:
            project=server.project(projectPath)
            experiments=[project.byFullPath(e) for e in workPerProject[projectPath]]
            reporter.task("Launching:" + str(len(experiments)), len(experiments))

            allTasks=[]
            tn=[]
            for i in self.tasks:
                if i=="all":
                    tn=tn+[t.name for t in project.get_tasks()]
                else:
                    tn.append(i)
            for exp in experiments:
                for task in tn:
                    t = project.get_task_by_name(task)
                    allTasks=allTasks+t.createTodo(exp,self.taskArgs,project)
            executor.execute(allTasks)
            for x in allTasks:
                err=None
                if x.exception is not None:
                    err=x.exception.log()
                if hasattr(x,"results"):
                    rs.append({"results":x.results,"exception":err})
                else:
                    rs.append({"results": None, "exception": err})

        reporter.done()
        return rs


class ValidateModel(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.dside.ui.ModelEvaluationSpec'

    def __init__(self, **kwargs):
        pass

class ModelSpec(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.dside.ui.ModelEvaluationSpec'

    def __init__(self,**kwargs):
        self.args=kwargs
        self.folds="ALL"
        self.seeds = "ALL"
        self.hasSeeds=False
        self.stages="LAST_STAGE"
        self.folds_numbers=[]
        self.stages_numbers = []
        self.seed_numbers = []
        for v in kwargs:
            setattr(self, v, kwargs[v])
        pass

    def wrap(self,m:GenericTaskConfig,e)->model.ConnectedModel:
        f=e.concrete()
        if len(f)>1 and self.hasSeeds:
            if self.seeds=="ALL":
                ch=[self.wrap(x.parse_config(),x) for x in f]
                return model.AverageBlend(ch)
            if self.seeds == "MANUAL":
                f = [f[x] for x in self.seed_numbers]
                ch = [self.wrap(x.parse_config(), x) for x in f]
                if len(ch)==1:
                    return ch[0]
                return model.AverageBlend(ch)
                pass
        folds=list(range(m.folds_count))
        stages=len(m.stages)-1
        if self.folds=="MANUAL":
            folds=self.folds_numbers
            pass
        if self.stages=="MANUAL":
            folds=self.stages_numbers
            pass
        if self.stages=="ALL":
            stages=list(range(m.stages))
            pass
        return model.FoldsAndStages(m,folds,stages)



class Introspect(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.musket_core.IntrospectTask'

    def __init__(self,path,outPath=None):
        self.path=path;
        self.outPath=outPath
        pass

    def perform(self, server, reporter: ProgressMonitor):
        project = server.project(self.path)
        r=project.introspect()
        if self.outPath is not None:
            save_yaml(self.outPath,r)
        return r

class RunJavaServer(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.dside.tasks.GateWayRelatedTask'

    def __init__(self):
        self.listeningPort=2
        pass

    def perform(self, server, reporter: ProgressMonitor):
        java_server.launch(self.listeningPort)


def _exactValue(x,y):
    allCorrect=np.equal(x>0.5,y>0.5).sum()==len(x)
    if allCorrect:
        return "Correct"
    return "Incorrect"


class WrappedDS(datasets.SubDataSet):

    def __init__(self,orig,indexes,name,visualizer=None):
        super().__init__(orig,indexes)
        self._visualizer=visualizer
        self._name=name
    def len(self):
        return len(self)

    def config(self):
        return ""

    def item(self,num):
        return self._visualizer[num]

    def name(self):
        return self._name

    class Java:
        implements = ["com.onpositive.musket_core.IDataSet"]
    pass

class AnalizeResults:
    def __init__(self,ds):
        self._results=ds
        pass

    def get(self,index):
        return self._results[index]

    def size(self):
        return len(self._results)

    def names(self):
        return [x.name() for x in self._results]

    class Java:
        implements = ["com.onpositive.dside.tasks.analize.IAnalizeResults"]



class AnalizeOptionsRequest(yaml.YAMLObject):

    yaml_tag = u'!com.onpositive.dside.dto.GetPossibleAnalisisInfo'

    def __init__(self,**kwargs):
        self.spec=None
        self.experimentPath=None
        self.datasetName=None
        self.metricFunction=_exactValue
        pass

    def perform(self, server, reporter: ProgressMonitor):
        exp:Experiment=server.experiment(self.experimentPath)
        project:projects.Project=exp.project
        rs={
            "visualizers":[v.introspect() for v in project.get_visualizers()],
            "analizers": [v.introspect() for v in project.get_analizers()],
            "data_analizers": [v.introspect() for v in project.get_data_analizers()]
        }
        return yaml.dump(rs)



class AnalizePredictions(yaml.YAMLObject):

    yaml_tag = u'!com.onpositive.dside.dto.DataSetAnalisysRequest'

    def __init__(self,**kwargs):
        self.spec=None
        self.experimentPath=None
        self.datasetName=None
        self.analizer=_exactValue
        self.visualizer=None
        self.data=False
        pass

    def perform(self, server, reporter: ProgressMonitor):
        ms=ModelSpec(**self.spec)
        exp:Experiment=server.experiment(self.experimentPath)
        cf=exp.parse_config()

        ds=cf.get_dataset(self.datasetName)
        wrappedModel = ms.wrap(cf, exp)

        targets=datasets.get_targets_as_array(ds)
        analizerFunc = exp.project.get_analizer_by_name(self.analizer).clazz
        visualizerFunc = exp.project.get_visualizer_by_name(self.visualizer)
        if self.data:
            pass
            l = len(targets)
            res = {}
            for i in tqdm(range(l)):
                gt = targets[i]
                gr = analizerFunc(gt)
                if gr in res:
                    res[gr].append(i)
                else:
                    res[gr] = [i]
        else:
            predictions=wrappedModel.predictions(self.datasetName)
            l=len(targets)
            res={}
            for i in tqdm(range(l)):
                gt=targets[i]
                pr=predictions[i]
                gr=analizerFunc(gt,pr)
                if gr in res:
                    res[gr].append(i)
                else:
                    res[gr]=[i]

        _results=[]
        for q in res:
            r=WrappedDS(ds,res[q],str(q))
            r._visualizer=visualizerFunc.create(r,tempfile.mkdtemp())
            _results.append(r)
        return AnalizeResults(_results)


class Launch(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.dside.ui.LaunchConfiguration'

    def __init__(self,gpusPerNet,numGpus,numWorkers,experiments,allowResume=False,onlyReports:bool=False,launchTasks:bool=False):
        self.gpusPerNet=gpusPerNet
        self.numGpus=numGpus
        self.numWorkers=numWorkers
        self.experiments=experiments
        self.allowResume=allowResume
        self.onlyReports=onlyReports
        self.launchTasks=launchTasks
        pass




    def perform(self,server,reporter:ProgressMonitor):
        workPerProject={}
        for e in self.experiments:
            inde=e.index("experiments")
            pp=e[0:inde]
            localPath=e
            if pp in workPerProject:
                workPerProject[pp].append(localPath)
            else:
                workPerProject[pp]=[localPath]
        executor = parralel.get_executor(self.numWorkers, self.numGpus)

        for projectPath in workPerProject:
            project=server.project(projectPath)
            experiments=[project.byFullPath(e) for e in workPerProject[projectPath]]
            reporter.task("Launching:" + str(len(experiments)), len(experiments))
            allTasks=[]
            for exp in experiments:

                exp.allowResume=self.allowResume
                exp.gpus=self.gpusPerNet
                exp.onlyReports=self.onlyReports
                exp.launchTasks=self.launchTasks
                if exp.hyperparameters() is not None:
                    hyper.optimize(exp,executor,reporter)

                else:
                    try:
                        allTasks=allTasks+exp.fit(reporter)
                    except:
                        pass
                    if len(allTasks)>self.numWorkers:
                        executor.execute(allTasks)
                        allTasks=[]
            executor.execute(allTasks)
        reporter.done()