from py4j.clientserver import ClientServer, JavaParameters, PythonParameters,JavaGateway,GatewayParameters,CallbackServerParameters
from musket_core import projects, tools, parralel
import yaml
import io
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api_client import ApiClient

import json

class DataSetProxy:
    def __init__(self,ds:projects.WrappedDataSet,p):
        self.p=p
        self.ds=ds

        pass

    def len(self):
        return len(self.ds)

    def config(self):
        return self.ds.w.name+"("+str(self.ds.parameters)[1:-1]+")"

    def get_name(self):
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

def convert_ds(ds):
    return {"ref": ds.ref, "url": ds.url}

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

    def performTask(self,config:str,reporter:tools.ProgressMonitor):
        try:
            config=config[1:].replace("!!com.onpositive","!com.onpositive")
            obj=yaml.load(config)
            results=obj.perform(self.w,reporter)
            return results
        except:
            parralel.Error()

    def getDatasets(self, search, mine):
        api = KaggleApi(ApiClient())

        api.authenticate()

        return json.JSONEncoder().encode([convert_ds(item) for item in api.dataset_list(search=search, mine=mine)])

    def downloadDataset(self, id, dest):
        api = KaggleApi(ApiClient())

        api.authenticate()

        api.dataset_download_files(id, dest, quiet=False, unzip=True)

    class Java:
        implements = ["com.onpositive.musket_core.IServer"]


def launch(port):
    server = Server()
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=port),
        callback_server_parameters=CallbackServerParameters(port=0))
    python_port = gateway.get_callback_server().get_listening_port()
    gateway.start_callback_server()
    gateway.java_gateway_server.resetCallbackClient(
        gateway.java_gateway_server.getCallbackClient().getAddress(),
        python_port)
    gateway.entry_point.created(server)