from py4j.clientserver import ClientServer, JavaParameters, PythonParameters,JavaGateway
from musket_core import projects
import yaml
import io
from musket_core import tools
import sys

class DataSetProxy:
    def __init__(self,ds:projects.WrappedDataSet,p):
        self.p=p
        self.ds=ds

        pass

    def len(self):
        return len(self.ds)

    def config(self):
        return self.ds.w.name+"("+str(self.ds.parameters)[1:-1]+")"

    def name(self):
        return self.ds.name

    def item(self,num):
        return self.ds.visualizer()[num]

    class Java:
        implements = ["com.onpositive.musket_core.IDataSet"]

class List:

    def __init__(self,collection):
        self.collection=collection

    def get(self,index):
        return self.collection[index]

    def size(self):
        return len(self.collection)

    class Java:
        implements = ["com.onpositive.musket_core.IList"]


class Project(projects.Project):

    def __init__(self,path):
        self.path=path
        self.project=projects.Project(path)
        self.project.get_visualizers()

    def datasets(self):
        return List([DataSetProxy(x,self) for x in self.project.get_datasets()])

    def byFullPath(self,path):
        return self.project.byFullPath(path)

    class Java:
        implements = ["com.onpositive.musket_core.IProject"]


class Server(projects.Workspace):

    def __init__(self):
        super().__init__()
        self.w=projects.Workspace()
        pass

    def project(self,path):
        if path in self.projects:
            return self.projects[path]
        p=Project(path)
        self.projects[path]=p
        return p

    def performTask(self,config,reporter:tools.ProgressMonitor):
        obj=yaml.load(config[1:])
        wor=sys.stdout.write
        wer = sys.stderr.write
        def newWrite(t):
            wor(t)
            reporter.stdout(t)
        def newErr(t):
            wer(t)
            reporter.stdout(t)
        try:
            sys.stdout.write=newWrite
            sys.stderr.write=newErr
            results=obj.perform(self.w,reporter)
            return yaml.dump(results)
        finally:
            sys.stdout.write=wor
            sys.stderr.write=wer
            pass

    class Java:
        implements = ["com.onpositive.musket_core.IServer"]

server = Server()
gateway = JavaGateway(auto_convert=True,
            python_server_entry_point=server)
gateway.start_callback_server()
