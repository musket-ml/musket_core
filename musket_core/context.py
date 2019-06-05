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
    for frm in st:        
        file=frm.filename;        
        dn=os.path.dirname(file)
        while len(dn)>0 :
            if os.path.exists(os.path.join(dn,"modules")):
                return dn
            old=dn
            dn=os.path.dirname(dn)
            if old==dn:
                break  
    return None            
        

def get_current_project_path():
    return _find_path()
#     if not hasattr(context,"projectPath") and False:
#         context.projectPath=_find_path();
#     return context.projectPath


