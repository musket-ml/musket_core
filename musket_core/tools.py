from musket_core import parralel
from musket_core import hyper
from musket_core import model
from musket_core import datasets
from musket_core.generic_config import GenericTaskConfig
from musket_core.utils import save_yaml,ensure
from musket_core import projects
from musket_core.experiment import Experiment
from musket_core import dataset_analizers
from musket_core import context
import numpy as np
import tempfile
from tqdm import tqdm
import inspect
import os
from musket_core.datasets import DataSet, SubDataSet

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
        try:
            import segmentation_models
            
        except:
            pass    
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


class AnalizeResults:
    def __init__(self,ds,visualSpec=None):
        self._results=ds
        self._visualSpec=visualSpec
        pass

    def get(self,index):
        return self._results[index]

    def size(self):
        return len(self._results)

    def names(self):
        return [x.name() for x in self._results]

    class Java:
        implements = ["com.onpositive.dside.tasks.analize.IAnalizeResults"]

    def visualizationSpec(self):
        return yaml.dump(self._visualSpec)


class LastDataSetCache:

    def __init__(self):
        self._ds=None
        self._cfg=None
        self._config=None
        self._targets=None
        pass

    def get_dataset(self,e:Experiment,name:str)->datasets.DataSet:
        if self._cfg==e.path+":"+name:
            if self._ds is not None:
                return self._ds

        self._config=e.parse_config()
        self._ds=self._config.get_dataset(name)
        self._cfg = e.path + ":" + name
        return self._ds

    def get_targets(self,e:Experiment,name:str)->datasets.DataSet:
        if self._cfg==e.path+":"+name:
            if self._targets is not None:
                return self._targets

        self._targets = datasets.get_targets_as_array(self.get_dataset(e,name))
        return self._targets

_cache=LastDataSetCache()


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
        print("Aquiring dataset...")
        ds=_cache.get_dataset(exp,self.datasetName)

        project:projects.Project=exp.project

        rs={
            "visualizers":[v.introspect() for v in project.get_visualizers()],
            "analizers": [v.introspect() for v in project.get_analizers() if v.isApplicable(ds)],
            "data_analizers": [v.introspect() for v in project.get_data_analizers() if v.isApplicable(ds)],
            "datasetStages": datasets.get_stages(ds),
            "datasetFilters":[v.introspect() for v in project.get_data_filters()],
        }
        return yaml.dump(rs)
    
class ExportResultsRequest(yaml.YAMLObject):

    yaml_tag = u'!com.onpositive.dside.dto.ExportDataSet'

    def __init__(self,**kwargs):
        self.spec=None
        self.experimentPath=None
        self.datasetName=None
        self.metricFunction=_exactValue
        pass

    def perform(self, server, reporter: ProgressMonitor):
        exp:Experiment=server.experiment(self.experimentPath)
        ms=ModelSpec(**self.spec)
        cf=exp.parse_config()
        wrappedModel = ms.wrap(cf, exp)
        predictions=wrappedModel.predictions(self.datasetName);
        ps=str(wrappedModel.stages)+"."+str(wrappedModel.folds)
        parentPath=os.path.join(os.path.dirname(cf.path),"predictions")
        ensure(parentPath)
        p1=os.path.join(parentPath,self.datasetName+ps+"-pr.csv")
        predictions.dump(p1)
        if self.exportGroundTruth:
            p2=os.path.join(parentPath,self.datasetName+ps+"-gt.csv")
            predictions.dump(p2,encode_y=True)
            return p1+"::::"+p2
        return p1



class AnalizePredictions(yaml.YAMLObject):

    yaml_tag = u'!com.onpositive.dside.dto.DataSetAnalisysRequest'

    def __init__(self,**kwargs):
        self.spec=None
        self.experimentPath=None
        self.datasetName=None
        self.analizer=_exactValue
        self.visualizer=None
        self.data=False
        self.analzierArgs={}
        self.visualizerArgs={}
        self.stage=None
        self.filters=[]
        pass

    def accept_filter(self,ds:datasets.DataSet,i:int):
        if self.filters is None or len(self.filters)==0:
            return True

        for x in self.filters:
            f=x[0]
            d=x[1]
            a=x[2]

            item = d[i]
            if not f(item,a):
                return False

        return True


    def perform(self, server, reporter: ProgressMonitor):
        ms=ModelSpec(**self.spec)
        exp:Experiment=server.experiment(self.experimentPath)
        cf=exp.parse_config()

        ds = _cache.get_dataset(exp, self.datasetName)
        if self.stage is not None:
            ds=datasets.get_stage(ds,self.stage)
        wrappedModel = ms.wrap(cf, exp)

        analizer_by_name = exp.project.get_analizer_by_name(self.analizer)

        analizerFunc = analizer_by_name.clazz
        isClass=False
        if (analizer_by_name.isClass):
            isClass=True
            analizerFunc=analizerFunc(**self.analzierArgs)
            self.analzierArgs={}
            analizerFunc.usePredictionItem=True
        visualizerFunc = exp.project.get_visualizer_by_name(self.visualizer)
        targets=_cache.get_targets(exp,self.datasetName)
        filters=[]
        for f in self.filters:
            fa:dict=f
            flt_func=exp.project.get_filter_by_name(fa["filterKind"]).clazz
            if flt_func==dataset_analizers.custom_python:
                d={}
                r = exec(fa["filterArgs"], d, d) in d
                for m in d:
                    if inspect.isfunction(d[m]):
                        r = d[m]

                        def fff(x, args):
                            return r(x)

                        flt_func = fff
            if fa["mode"]=="inverse":
                def inv(x,a):
                    return not flt_func(x,a)
                flt=inv
            else:
                flt=flt_func
            m=_cache.get_dataset(exp, self.datasetName)
            fltDat=datasets.get_stage(m,fa["applyAt"])
            filterArgs=fa["filterArgs"]
            filters.append((flt,fltDat,filterArgs))
        self.filters=filters
        predictions=None
        if self.data:
            pass
            l = len(targets)
            res = {}
            for i in tqdm(range(l)):
                gt = targets[i]
                if not self.accept_filter(ds,i):
                    continue
                if analizerFunc.usePredictionItem:
                    gr = analizerFunc(i,ds[i], **self.analzierArgs)
                else: gr = analizerFunc(gt,**self.analzierArgs)
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
                if not self.accept_filter(ds,i):
                    continue
                pr=predictions[i]
                if analizerFunc.usePredictionItem:
                    gr = analizerFunc(i,ds[i],pr, **self.analzierArgs)
                else: gr=analizerFunc(pr.y,pr.prediction,**self.analzierArgs)
                if gr in res:
                    res[gr].append(i)
                else:
                    res[gr]=[i]

        _results=[]
        visualizationHints=None
        if isClass:
            res=analizerFunc.results()
            visualizationHints=analizerFunc.visualizationHints()
        for q in res:
            if isinstance(res[q],DataSet):
                r=WrappedDS(res[q],list(range(len(res[q]))),str(q),None,predictions)
            else: r=WrappedDS(ds,res[q],str(q),None,predictions)
            r._visualizer=visualizerFunc.create(r,tempfile.mkdtemp())
            if (len(self.visualizerArgs)) > 0:
                r._visualizer.args=self.visualizerArgs
            _results.append(r)
        return AnalizeResults(_results,visualizationHints)

class Validate(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.musket_core.ValidateTask'

    def __init__(self):
        self.path=None

    def perform(self,server,reporter:ProgressMonitor):
        path=self.path
        e:Experiment= server.experiment(path)
        e.parse_config().validate()
        print("Model validated successfully!")
        return None
    
class ExportForWeb(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.musket_core.ExportForWeb'

    def __init__(self):
        self.path=None
        self.resultPath=None

    def perform(self,server,reporter:ProgressMonitor):
        path=self.path
        e:Experiment= server.experiment(path)
        files=os.listdir(path)
        print("Preparing for export as web service...")
        
        lm=os.path.join(path,"config.yaml.ncx")
        cfg=e.parse_config()
        context.context.projectPath=cfg.get_project_path()
        rp=os.path.join(self.resultPath,"assets");
        if not os.path.exists(lm):
            cfg.createNet()
        import shutil
        shutil.copy(lm, rp)    
        ds=cfg.get_dataset()
        stages=datasets.get_preprocessors(ds)
        lastDeploy=None
        for s in stages:
            if hasattr(s, "func"):
                func=s.func
                if hasattr(func, "deployHandler"):
                    func.deployHandler(s,cfg,self.resultPath)
            if hasattr(s, "deployHandler"):
                lastDeploy=s
                
        args=lastDeploy.deployHandler(cfg,self.resultPath)
        lm=os.path.join(path,"init.py")
        with open(lm, "w",encoding="utf-8") as f:
            f.write("""
from musket_core import inference
import os

@inference.inference_service_factory
def createEngine():
    """+args)                            
        print("Done.")    
        return None
        

class Launch(yaml.YAMLObject):
    yaml_tag = u'!com.onpositive.dside.ui.LaunchConfiguration'

    def __init__(self,gpusPerNet,numGpus,numWorkers,experiments,allowResume=False,onlyReports:bool=False,launchTasks:bool=False,folds=None, time=-1):
        self.gpusPerNet=gpusPerNet
        self.numGpus=numGpus
        self.numWorkers=numWorkers
        self.experiments=experiments
        self.allowResume=allowResume
        self.onlyReports=onlyReports
        self.launchTasks=launchTasks
        self.folds=folds
        self.time = time
        pass



    def perform(self,server,reporter:ProgressMonitor):
        if hasattr(self, 'folds') and self.folds is not None:
            if isinstance(self.folds, str):
                if len(self.folds)==0:
                    self.folds=None
                else:
                    self.folds=[int (x.strip()) for x in self.folds.split(",")]    
        #print(self.fold_numbers)
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
                if hasattr(self, "time"):
                    exp.time = self.time

                if hasattr(self, "folds"):
                    exp.folds=self.folds

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


class WrappedDS(SubDataSet):

    def __init__(self,orig,indexes,name,visualizer,predictions):
        super().__init__(orig,indexes)
        self._visualizer=visualizer
        self.name=name
        self.predictions=predictions
    def len(self):
        return len(self)

    def config(self):
        return ""

    def get_name(self):
        return self.name

    def item(self,num):
        return self._visualizer[num]
    
    def id(self,num):
        return self[num].id

    def __getitem__(self, item):
        it = super().__getitem__(item)
        if self.predictions is not None:
            if isinstance(item, slice):
                preds = [self.predictions[i] for i in self.indexes[item]]
                for i in range(len(preds)):
                    it[i].prediction = preds[i]
            else:
                it.prediction = self.predictions[self.indexes[item]].prediction
        return it

    class Java:
        implements = ["com.onpositive.musket_core.IDataSet"]
    pass
