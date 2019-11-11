from py4j.clientserver import ClientServer, JavaParameters, PythonParameters,JavaGateway,GatewayParameters,CallbackServerParameters
from musket_core import projects, tools, parralel, deps_download
import yaml
import io
import sys
import shutil
import zipfile
import os

from musket_core.kaggle_train_runner import loop, kernel

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.api_client import ApiClient
except:
    print("can't import kaggle!")

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
        implements = ["com.onpositive.musket_core.IMusketProject"]

def convert_ds(ds):
    return {"ref": ds.ref, "size": (ds.size if hasattr(ds, "size") else "unknown")}

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
            for d in dir(tools):
                clz=getattr(tools, d);
                if hasattr(clz, "yaml_tag"):
                    tg=getattr(clz, "yaml_tag");
                    ld=getattr(clz, "from_yaml")
                    yaml.FullLoader.add_constructor(tg, ld)                
            obj=yaml.load(config,Loader=yaml.FullLoader)
            results=obj.perform(self.w,reporter)
            return results
        except:
            parralel.Error()

    def getDatasets(self, search, mine):
        api = KaggleApi(ApiClient())

        api.authenticate()

        return json.JSONEncoder().encode([convert_ds(item) for item in api.dataset_list(search=search, mine=mine)])

    def getCompetitions(self, search, mine):
        api = KaggleApi(ApiClient())

        api.authenticate()

        return json.JSONEncoder().encode([convert_ds(item) for item in api.competitions_list("entered" if mine else "general", "all", "recentlyCreated", 1, search)])

    def downloadDataset(self, id, dest):
        api = KaggleApi(ApiClient())

        api.authenticate()

        api.dataset_download_files(id, dest, quiet=False, unzip=True)

        print("download complete")

    def downloadCompetitionFiles(self, id, dest):
        api = KaggleApi(ApiClient())

        api.authenticate()

        api.competition_download_files(id, dest, True, False)

        for item in os.listdir(dest):
            path = os.path.join(dest, item)

            if zipfile.is_zipfile(path):
                new_dir = path[0: path.rindex(".")]

                with zipfile.ZipFile(path) as zip:
                    zip.extractall(new_dir)

                    print("removing: " + path)

                os.remove(path)

        print("download complete")

    def runOnKaggle(self, projectPath):
        loop.MainLoop(kernel.Project(projectPath)).start()

    def downloadDeps(self, fullPath):
        print("downloading dependencies...")

        deps_download.download(fullPath)

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