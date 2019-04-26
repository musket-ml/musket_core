from threading import local
import inspect
from musket_core import utils
import os

context=local()

def _find_path():
    last=-1
    st=inspect.stack()
    for frm in st:        
        file=frm.filename;        
        dn=os.path.dirname(file)
        if last==0:
            last=last+1
            continue
        if last==1:
            return os.path.dirname(dn)  
        if os.path.basename(file)=="projects.py" and "musket_core" in dn:
            last=0
        

def get_current_project_path():
    if not hasattr(context,"projectPath"):
        context.projectPath=_find_path();
    return context.projectPath


