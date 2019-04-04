import keras
import inspect
from  keras.layers import Layer
from  keras.callbacks import Callback

def parameters(sig):
    if hasattr(sig, "original"):
       sig=getattr(sig, "original")
    cs = inspect.signature(sig)
    
    pars=[]
    for v in cs.parameters:
        parameter = cs.parameters[v]
        p={}
        if v=="self":
            continue
        if v=="args":
            continue
        if v=="kwargs":
            continue
        if parameter.kind==inspect._ParameterKind.POSITIONAL_OR_KEYWORD:
            p["kind"]="any"
        if parameter.kind == inspect._ParameterKind.KEYWORD_ONLY:
                p["kind"] = "keyword"
        if parameter.kind == inspect._ParameterKind.POSITIONAL_ONLY:
                p["kind"] = "positional"
        if parameter.annotation != inspect._empty:
            p["type"] = parameter.annotation.__name__
        p["name"]=v
        if parameter.default!=inspect._empty:
            p["defaultValue"]=str(parameter.default)
        pars.append(p)
    return pars
def record(m,kind):
    rs={}

    rs["doc"]=inspect.getdoc(m)
    if hasattr(m,"__name__"):
        rs["name"]=getattr(m,"__name__")

    if inspect.isclass(m):
        if hasattr(m,"__init__"):
            constructor=getattr(m,"__init__")
            rs["parameters"]=parameters(constructor)
    else:
        rs["parameters"]=parameters(m)     
    rs["kind"]=kind
    rs["sourcefile"]=inspect.getsourcefile(m)
    rs["source"] = inspect.getsource(m)
    return rs

blackList={'get','deserialize','deserialize_keras_object','serialize','serialize_keras_object','Layer','Callback','Optimizer'}
def instrospect(m,clazz):
    d = dir(m)
    l=[]
    for c in d:
        if c[0]=='_':
            continue
        if c in blackList:
            continue
        v=getattr(m,c)

        if inspect.isclass(v):
            if issubclass(v,clazz):
                l.append(record(v,clazz.__name__))
        if inspect.isfunction(v) and isinstance(clazz,str):
            l.append(record(v, clazz))
    return l

losses=instrospect(keras.losses,"losses")
metrics=instrospect(keras.metrics,"metrics")
optimizer=instrospect(keras.optimizers,keras.optimizers.Optimizer)
layers=instrospect(keras.layers,Layer)
callbacks=instrospect(keras.callbacks,Callback)

def builtins():
    return losses+metrics+optimizer+layers+callbacks
