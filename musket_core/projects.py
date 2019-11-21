from musket_core import experiment,structure_constants, builtin_datasets
from musket_core.context import context
from musket_core import datasets,visualization,utils
from musket_core import parralel
from musket_core import dataset_analizers,dataset_visualizers,metrics
from musket_core import crf
import keras
import  musket_core.model
from typing import Collection
import os
import sys

import importlib
import inspect

from musket_core import introspector



def _all_experiments(path,e:[experiment.Experiment]):
    if structure_constants.isExperimentDir(path):
        e.append(experiment.Experiment(path))
        return
    for d in os.listdir(path):
        fp=os.path.join(path, d)
        if os.path.isdir(fp):
            _all_experiments(fp,e)


class WrappedDataSetFactory:
    def __init__(self,name,func,sig,project):
        self.name=name
        self.sig=sig
        self.func=func
        self.project=project

    def __str__(self):
        return self.name+str(self.sig)

    def __hash__(self):
        return hash(str(self))

    def create(self,name, parameters):
        return WrappedDataSet(name,self, parameters,self.project)

    def introspect(self):
        r={}
        r["name"]=self.name
        r["kind"]="dataset_factory"
        r["parameters"]=introspector.parameters(self.func)
        try:
            r["sourcefile"] = inspect.getsourcefile(self.func)
            r["source"] = inspect.getsource(self.func)
        except:
            pass
        return r

class WrappedVisualizer:
    def __init__(self,name,func,sig):
        self.name=name
        self.sig=sig
        self.func=func

    def __str__(self):
        return self.name+str(self.sig)

    def __hash__(self):
        return hash(str(self))

    def create(self,d:datasets.DataSet,path):
        utils.ensure(path)
        return visualization.Visualizer(self.func,path,d)

    def introspect(self):
        r = {}
        r["name"] = self.name
        r["kind"] = "visualizer"
        r["parameters"] = introspector.parameters(self.func)
        if hasattr(self.func,"viewer"):
            r["viewer"]=getattr(self.func,"viewer")
        else:
            r["viewer"] = "text"
        try:
            r["sourcefile"] = inspect.getsourcefile(self.func)
            r["source"] = inspect.getsource(self.func)
        except:
            pass
        return r

class WrappedTask:
    def __init__(self,name,func,sig):
        self.name=name
        self.sig=sig
        self.func=func

    def __str__(self):
        return self.name+str(self.sig)

    def __hash__(self):
        return hash(str(self))

    def createConcreteTodo(self, exp, args, project):
        def executeTask(t):
            actualArgs={}
            config=exp.parse_config()
            context.projectPath=project.path
            for p in self.sig.parameters:
                par=self.sig.parameters[p]
                type=par.annotation
                if issubclass(type,musket_core.model.Model):
                    if p in args:
                        actualArgs[p]=args[p].wrap(config,exp)
                    else: actualArgs[p]=config

                elif issubclass(type,musket_core.datasets.DataSet):
                    actualArgs[p]=config.get_dataset(args[p])
                    if not hasattr(actualArgs[p],"name") and not hasattr(actualArgs[p],"origName"):
                        actualArgs[p].name=args[p]
                    pass
                else:
                    if p in args:
                        actualArgs[p]=args[p]
            taskFolder=os.path.join(os.path.dirname(config.path),self.name)
            utils.ensure(taskFolder)
            os.chdir(taskFolder)

            self.func(**actualArgs)
            t.results=taskFolder
        return parralel.Task(executeTask,True,name=self.name,needs_tasks=True)

    def createTodo(self,exp,args,project):
        if args==None:
            args={}
        concrete=exp.concrete()
        return [self.createConcreteTodo(v,args,project) for v in concrete]

    def introspect(self):
        r={}
        r["name"]=self.name
        r["kind"]="task"
        r["parameters"]=introspector.parameters(self.func)
        try:
            r["sourcefile"]=inspect.getsourcefile(self.func)
            r["source"] = inspect.getsource(self.func)
        except:
            pass
        return r

class WrappedLayer:
    def __init__(self,clazz):
        self.name=clazz.__name__
        self.clazz=clazz


    def introspect(self):
        return  introspector.record(self.clazz,"Layer")

class WrappedPreprocessor:
    def __init__(self,clazz):
        self.name=clazz.__name__
        self.clazz=clazz


    def introspect(self):
        return  introspector.record(self.clazz,"preprocessor")

class WrappedModelBlock:
    def __init__(self,clazz):
        self.name=clazz.__name__
        self.clazz=clazz

    def introspect(self):
        return  introspector.record(self.clazz,"model")

class WrappedAnalizer:
    def __init__(self,clazz):
        self.name=clazz.__name__
        self.clazz=clazz
        self.isClass=inspect.isclass(clazz)
        
    def isApplicable(self,dataset):
        if hasattr(self.clazz, "checker"):
            return self.clazz.checker(dataset)
        return True    

    def introspect(self):
        return  introspector.record(self.clazz,"analizer")

class WrappedDataAnalizer:
    def __init__(self,clazz):
        self.name=clazz.__name__
        self.isClass = inspect.isclass(clazz)
        self.clazz=clazz
        
    def isApplicable(self,dataset):
        if hasattr(self.clazz, "checker"):
            return self.clazz.checker(dataset)
        return True

    def introspect(self):
        return  introspector.record(self.clazz,"data_analizer")

class WrappedDataFilter:
    def __init__(self,clazz):
        self.name=clazz.__name__
        self.clazz=clazz

    def introspect(self):
        return  introspector.record(self.clazz,"data_filter")

class WrappedDataSet(datasets.DataSet):

    def __init__(self,name,w:WrappedDataSetFactory,parameters,project):
        super().__init__()
        self.w=w
        self.name=name
        self.project=project
        self.parameters=parameters
        self._inner_=None
        self._vis=None

    def __getitem__(self, item):
        return self.inner()[item]

    def visualizer(self):
        if self._vis is not None:
            return self._vis
        vs=self.project.get_visualizers()[0]
        vz=self.project._attach_visualizer(vs,self)
        self._vis=vz
        return vz

    def inner(self):
        if self._inner_ is not None:
            return self._inner_

        self._inner_=self.w.func(*self.parameters)
        return self._inner_

    def __hash__(self):
        return hash(str(self))+hash(self.w)

    def __eq__(self, other):
        if str(self)==str(other):
            if isinstance(other,WrappedDataSet):
                return str(self.w)==str(other.w)
        return False

    def __len__(self):
        return len(self.inner())

    def __str__(self):
        return self.name+str(self.parameters)
    def __repr__(self):
        return str(self)

class Workspace:

    def __init__(self):
        self.projects = {}
        pass

    def project(self, path):
        if path in self.projects:
            return self.projects[path]
        p = Project(path)
        self.projects[path] = p
        return p

    def experiment(self,experimentPath):
        project = self.project(experimentPath[0:experimentPath.index("experiments")])
        exp = project.byFullPath(experimentPath)
        return exp

class Project:

    def __init__(self,path):
        self.path=path
        sys.path.insert(0, self.modulesPath())
        self.__module_cache={}
        self._elements=None
        self._experiments=None
        pass


    def experiments(self)->Collection[experiment.Experiment]:
        if self._experiments is not None:
            return self._experiments
        result=[]
        _all_experiments(self.experimentsPath(),result)
        for e in result:
            e.project=self
        self._experiments=result
        return result


    def byName(self,name):
        ep=os.path.join(self.path, "experiments",name)
        if structure_constants.isExperimentDir(ep):
            return experiment.Experiment(ep,project=self)
        return None

    def byFullPath(self,name):
        if structure_constants.isExperimentDir(name):
            return experiment.Experiment(name,project=self)
        return None


    def experimentsPath(self):
        return os.path.join(self.path,"experiments")

    def modulesPath(self):
        return os.path.join(self.path,"modules")

    def dataPath(self):
        return os.path.join(self.path, "data")

    def commonPath(self):
        return os.path.join(self.path,"common.yaml")

    def modules(self):

        res=[]
        for m in os.listdir(self.modulesPath()):
            if m.endswith(".py"):
                x=m[:-3]
                res.append(self.module(x))
        res.append(dataset_analizers)
        res.append(dataset_visualizers)
        res.append(builtin_datasets)
        res.append(metrics)
        res.append(crf)
        if os.path.exists(self.commonPath()):
            
                mods=utils.load_yaml(self.commonPath())
                if mods is not None:                
                    if "imports" in mods:
                        if isinstance(mods["imports"],list):
                            for m in mods["imports"]:
                                res.append(self.module(m))
        try:
            res.append(self.module("musket_text.preprocessors"))
        except:
            pass
                                    
        return res

    def module(self,name):
        if name in self.__module_cache:
            return self.__module_cache[name]
        self.__module_cache[name]=importlib.import_module(name)
        return self.__module_cache[name]

    def introspect(self):
        self._elements=None
        for x in self.__module_cache:
            importlib.reload(self.__module_cache[x])
        res=[x.introspect() for x in self.elements()]
        for x in res:
            x["custom"]=True
        res=res+introspector.builtins()
        return {"features":res}

    def elements(self):
        if self._elements is not None:
            return self._elements
        self._elements=self._introspect()
        return self._elements

    def element(self,name):
        for i in self.elements():
            if i.name==name:
                return i
        return None

    def _introspect(self):
        elements=[]
        for m in self.modules():
            z=dir(m)
            for name in z:
                if name[0]=='_':
                    continue
                vl=getattr(m,name)
                if inspect.isfunction(vl):
                    sig=inspect.signature(vl)
                    if sig.return_annotation is not None:
                        d=sig.return_annotation
                        try:
                            if d==datasets.DataSet or issubclass(d,datasets.DataSet):
                                elements.append(WrappedDataSetFactory(name,vl,sig,self))
                        except:
                          pass
                    if hasattr(vl,"visualizer") and getattr(vl,"visualizer")==True:
                        elements.append(WrappedVisualizer(name,vl,sig))
                    if hasattr(vl,"dataset") and getattr(vl,"dataset")==True:
                        elements.append(WrappedDataSetFactory(name, vl, sig, self))
                    if hasattr(vl,"task") and getattr(vl,"task")==True:
                        elements.append(WrappedTask(name, vl, sig))
                    if hasattr(vl,"model") and getattr(vl,"model")==True:
                        elements.append(WrappedModelBlock(vl))
                    if hasattr(vl,"analizer") and getattr(vl,"analizer")==True: 
                        elements.append(WrappedAnalizer(vl))
                    if hasattr(vl,"data_analizer") and getattr(vl,"data_analizer")==True:
                        elements.append(WrappedDataAnalizer(vl))
                    if hasattr(vl, "preprocessor") and getattr(vl, "preprocessor") == True:
                        elements.append(WrappedPreprocessor(vl))
                    if hasattr(vl, "data_filter") and getattr(vl, "data_filter") == True:
                        elements.append(WrappedDataFilter(vl))
                if inspect.isclass(vl):
                    if hasattr(vl,"analizer") and getattr(vl,"analizer")==True: 
                        elements.append(WrappedAnalizer(vl))
                    if hasattr(vl,"data_analizer") and getattr(vl,"data_analizer")==True:
                        elements.append(WrappedDataAnalizer(vl))
                    if issubclass(vl,keras.layers.Layer):
                        file=inspect.getsourcefile(vl)
                        if not "keras" in file:
                            elements.append(WrappedLayer(vl))

        return elements

    def get_visualizers(self):
        return [x for x in self.elements() if isinstance(x,WrappedVisualizer)]

    def get_analizers(self):
        return [x for x in self.elements() if isinstance(x,WrappedAnalizer)]

    def get_data_analizers(self):
        return [x for x in self.elements() if isinstance(x,WrappedDataAnalizer)]

    def get_data_filters(self):
        return [x for x in self.elements() if isinstance(x,WrappedDataFilter)]

    def get_tasks(self):
        return [x for x in self.elements() if isinstance(x, WrappedTask)]

    def get_task_by_name(self,name):
        for t in self.get_tasks():
            if t.name==name:
                return t
        return None

    def get_visualizer_by_name(self,name)->WrappedVisualizer:
        for t in self.get_visualizers():
            if t.name==name:
                return t

    def get_analizer_by_name(self,name):
        for t in self.get_analizers():
            if t.name==name:
                return t
        for t in self.get_data_analizers():
            if t.name==name:
                return t
        return None

    def get_filter_by_name(self,name):
        for t in self.get_data_filters():
            if t.name==name:
                return t
        return None

    def get_datasets(self):
        ds=set()
        for e in self.experiments():
                cf=e.config()
                if "dataset" in cf:
                    d=cf["dataset"]
                    ds.add(self._create_dataset("dataset",d))

                if "datasets" in cf:
                    dss=cf["datasets"]
                    for v in dss:
                        ds.add(self._create_dataset(v,dss[v]))
        return ds

    def get_dataset(self,name):
        for d in self.get_datasets():
            if d.name==name:
                return d
        return None


    def _create_dataset(self,name,d):
        for v in d:
            wf=self.element(v)
            r=wf.create(name,d[v])
            return r


    def get_visualizer(self,name,datasetName)->visualization.Visualizer:
        visualization=os.path.join(self.path, "visualizations", name, datasetName)
        utils.ensure(visualization)
        return self.element(name).create(self.get_dataset(datasetName),visualization)

    def _attach_visualizer(self,visualizer,dataset)->visualization.Visualizer:
        visualization=os.path.join(self.path, "visualizations", visualizer.name, dataset.name)
        utils.ensure(visualization)
        return visualizer.create(dataset,visualization)