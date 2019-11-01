import keras

import inspect
from  keras.layers import Layer
from  keras.callbacks import Callback
from musket_core import net_declaration

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


def extra_params(parameters): 
    def inner(func): 
        func.extra_params=parameters
        return func        
    return inner #this is the fun_obj mentioned in the above content

def record(m,kind):
    rs={}
    if hasattr(m, "original"):
       m=getattr(m, "original")
    rs["doc"]=inspect.getdoc(m)
    if hasattr(m,"__name__"):
        rs["name"]=getattr(m,"__name__")

    if inspect.isclass(m):
        if hasattr(m,"__init__"):
            constructor=getattr(m,"__init__")
            rs["parameters"]=parameters(constructor)
            if kind=="Layer":
                rs["parameters"].append({"name":"trainable","type":"bool","defaultValue":"true",
                    })
    else:
        rs["parameters"]=parameters(m)     
    rs["kind"]=kind
    try:
        rs["sourcefile"]=inspect.getsourcefile(m)
        rs["source"] = inspect.getsource(m)
        if hasattr(m, "extra_params"):
            prms=getattr(m, "extra_params")
            if "parameters" in rs:
                rs["parameters"]=rs["parameters"]+prms
            else:
                rs["parameters"]=prms    
    except:
        pass
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
            try:
                if isinstance(clazz,type) and issubclass(v, clazz):
                    l.append(record(v,clazz.__name__))
            except:                
                print("Can not inspect",v)

        if inspect.isfunction(v) and isinstance(clazz,str):
            l.append(record(v, clazz))
    return l

losses=instrospect(keras.losses,"losses")
metrics=instrospect(keras.metrics,"metrics")
optimizer=instrospect(keras.optimizers,keras.optimizers.Optimizer)
layers=instrospect(keras.layers,Layer)
callbacks=instrospect(keras.callbacks,Callback)
bs=[]


def createPars(c:str):
    if "cache" in c: return [
        {

            "name": "split",
            "type": "bool",
            "defaultValue": True,
        }

    ]
    if "pass" == c: return []
    if c == "augmentation":
        return [
            {
                "name": "body",
                "type": "Preprocessor[]"
            },
            {
                "name": "weights",
                "type": "int[]"
            },
            {
                "name": "seed",
                "type": "int",
                "defaultValue": 0
            }
        ]
    if "preprocessor" in c:
        return [
            {

                "name": "body",
                "type": "Preprocessor[]"
            }

        ]
    return [
        {

            "name": "body",
            "type": "Layer[]"
        }

    ]


for c in net_declaration.builtins:
    if "preprocessor" in c:
        bs.append({"name": c, "kind": "preprocessor", "sourcefile": net_declaration.__file__,
                   "parameters": createPars(c)

                   })
    else:
        bs.append({ "name":c, "kind":"LayerOrPreprocessor","sourcefile":net_declaration.__file__,"parameters": createPars(c)


        })

objects = keras.utils.get_custom_objects()
for m in objects:
    if inspect.isfunction(objects[m]):
        bs.append(record(objects[m], "metric_or_loss"))
    else:
        record(objects[m],Layer)

def builtins():
    return losses+metrics+optimizer+layers+callbacks+bs
