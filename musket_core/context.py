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
        while len(dn)>0 :
            if os.path.exists(os.path.join(dn,"modules")):
                return dn
            old=dn
            dn=os.path.dirname(dn)
            if old==dn:
                break 
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
     
    return None            

def get_current_project_path():
    if not hasattr(context,"projectPath"):
        context.projectPath=_find_path();
    return context.projectPath

def get_current_project_data_path():
    if not hasattr(context,"projectPath"):
        context.projectPath=_find_path();
    return os.path.join(context.projectPath,"data")

def get_kaggle_input_root():
    is_on_kaggle = os.path.exists("/kaggle")

    if not is_on_kaggle:
        return get_current_project_data_path()

    data_name_path = "/kaggle/working/project/dataset_id.txt" if is_on_kaggle else os.path.join(get_current_project_data_path(), ".metadata/kaggle/project")

    input_path = "/kaggle/input" if is_on_kaggle else os.path.join(get_current_project_data_path(), "data")

    if os.path.exists(data_name_path):
        with open(data_name_path, "r") as f:
            new_input_path = os.path.join(input_path, f.read())

            if os.path.exists(new_input_path):
                return new_input_path

    return input_path

