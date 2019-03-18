from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
from musket_core import projects

class Server:
    def __init__(self):
        pass

    def hello(self):
        return "Hello"

    def project(self,path):
        return projects.Project(path)

    class Java:
        implements = ["com.onpositive.musket_core.IServer"]


server = Server()
gateway = ClientServer(
    java_parameters=JavaParameters(),
    python_parameters=PythonParameters(),
    python_server_entry_point=server)