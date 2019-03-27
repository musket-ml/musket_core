from musket_core import parralel
from musket_core import hyper
from musket_core.utils import save_yaml
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
                rs.append({"results":x.results,"exception":err})

        reporter.done()
        return rs


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