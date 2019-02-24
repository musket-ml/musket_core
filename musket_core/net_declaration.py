from musket_core import configloader
import keras
import musket_core.templating as tp
layers=configloader.load("layers")

import inspect
import tensorflow as tf

def take_input(layers,declarations,config,outputs,linputs,pName,withArgs):

    def a(args):
        return args
    return a

def seq(layers,declarations,config,outputs,linputs,pName,withArgs):

    layers=Layers(config,declarations,{},outputs,linputs,withArgs)
    return layers

def repeat(num):
    def repeat(layers,declarations,config,outputs,linputs,pName,withArgs):
        m=[]
        for v in range(num+1):
            cm=layers.parameters.copy()
            cm["_"]=v+1
            m.append(Layers(config, declarations, cm, outputs, linputs, withArgs))
        return m,num
    return repeat

def split(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m

def split_concat(layers, declarations, config, outputs, linputs, pName, withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Concatenate()

def split_add(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Add()

def split_substract(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Subtract()

def split_mult(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Multiply()

def split_min(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Minimum()

def split_max(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Maximum()

def split_dot(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Dot()

def split_dot_normalize(layers,declarations,config,outputs,linputs,pName,withArgs):
    m=[Layers([v], declarations, {}, outputs, linputs,withArgs) for v in config]
    return m,keras.layers.Dot(normalize=True)

builtins={
    "split": split,
    "split-concat": split_concat,
    "split-concatenate": split_concat,
    "split-add": split_add,
    "split-substract": split_substract,
    "split-mult": split_mult,
    "split-min": split_min,
    "split-max": split_max,
    "split-dot": split_dot,
    "split-dot-normalize": split_dot_normalize,
    "seq":seq,
    "input": take_input
}
for i in range(20):
    builtins["repeat("+str(i)+")"]=repeat(i)
gnum=0
class Layers:

    def __init__(self,layers_yaml,declarations,parameters,outputs=None,linputs=None,withArgs={}):
        global gnum
        layers_yaml=tp.resolveTemplates(layers_yaml,parameters)
        pName = "$input"
        self.layerMap:{str: keras.layers.Layer}={}
        self.layerInputs:{str:[str]} = {}
        self.layerArguments:{str:{str}}={}
        self.layerSequence:[keras.layers.Layer]=[]
        self.name="l"+str(gnum)
        self.parameters=parameters
        gnum=gnum+1
        nums={}
        for layer in layers_yaml:
            layerImpl=None
            key=list(layer.keys())[0]
            config = layer[key]
            isBuildin=False
            if key in builtins:
                layerImpl =builtins[key](self,declarations,config,outputs,linputs,pName,withArgs)
                if isinstance(layerImpl,list):

                    for i in layerImpl:
                        inputs = pName
                        name = i.name
                        self._add(config, inputs, i, name)
                        #pName = name
                        self.output = name
                        if outputs is not None:
                            self.output = outputs
                    pName = [i.name for i in layerImpl]
                    continue
                if isinstance(layerImpl,tuple):
                    second = layerImpl[1]
                    first=layerImpl[0]
                    if isinstance(second,int):
                        for i in first:
                            inputs = pName
                            name = i.name
                            self._add(config, inputs, i, name)
                            pName = name
                            self.output = name
                            if outputs is not None:
                                self.output = outputs
                        #pName = [i.name for i in first]
                    else:
                        for i in first:
                            inputs = pName
                            name = i.name
                            self._add(config, inputs, i, name)
                            #pName = name
                            self.output = name
                            if outputs is not None:
                                self.output = outputs
                        pName = [i.name for i in first]

                        second=layerImpl[1]
                        self._add({}, pName, second, second.name)
                        self.output = second.name
                        if outputs is not None:
                            self.output = outputs
                        pName=second.name
                    continue

                inputs = pName
                name = self.get_new_name(config, key, layerImpl, nums)
                isBuildin=True
            elif key in declarations:
                decl=declarations[key]
                layerImpl=decl.instantiate(declarations,config)
                inputs = config["inputs"] if "inputs" in config else pName
                name = self.get_new_name(config, key, layerImpl, nums)
            else:
                if config=="all":
                    layer[key]=[]
                layerImpl = layers.instantiate(layer, True,withArgs)[0]
                name = layer["name"] if "name" in layer else layerImpl.name
                if isinstance(config,dict):
                    inputs = config["inputs"] if "inputs" in config else pName
                else:
                    inputs=pName

            if inputs==pName:
                if isinstance(config,list) and not isBuildin:
                    all_refs=True
                    for v in config:
                        if v in self.layerMap or v in linputs:
                            pass
                        else:
                            all_refs=False
                    if all_refs:
                        inputs=config
                    pass
            self._add(config, inputs, layerImpl, name)
            pName=name
            self.output=name
            if outputs is not None:
                self.output=outputs
        pass

    def _add(self, config, inputs, layerImpl, name):
        self.layerMap[name] = layerImpl
        self.layerInputs[name] = inputs
        self.layerArguments[name] = config
        self.layerSequence.append(layerImpl)

    def get_new_name(self, config, key, layerImpl, nums):
        if key in nums:
            num = nums[key]
            nums[key] = nums[key] + 1
        else:
            num = 0
            nums[key] = 1

        if isinstance(config,dict) and "name" in config:
            name = config["name"]
        else:
            name = key + str(num)
        layerImpl.name = name
        return name

    def build(self,inputArgs):
        tensorMap={}
        last=None
        def findInput(name,n):
            if '[' in n:
                si=n.index('[')
                base=n[:si]
                quota=n.index(']')
                num=n[si+1:quota]
                return findInput(name,base)[int(num)]

            if isinstance(inputArgs,dict):
                if name in inputArgs:
                    return inputArgs[name]
                if n in inputArgs:
                    return inputArgs[n]
            if n in tensorMap:
                return tensorMap[n]

            if n=="$input":
                return inputArgs
            return None
        for l in self.layerSequence:
            inputs=self.layerInputs[l.name]
            if isinstance(inputs,str):
               inp=findInput(l.name,inputs)
               pass
            else:
               inp=[findInput(l.name,i) for i in inputs]
            if isinstance(inp,tuple) or isinstance(inp,list):
                if len(inp)==1:
                    inp=inp[0]
            if isinstance(inp,dict):
                if "$input" in inp and len(inp)==1:
                    inp=inp["$input"]
            res=l(inp)
            tensorMap[l.name]=res
            last=res
        if isinstance(self.output,str):
            return tensorMap[self.output]
        else:
            return [tensorMap[x] for x in self.output]

    def __call__(self, *args, **kwargs):
        return self.build(args)


class Declaration:

    def __init__(self,declaration_yaml):
        if isinstance(declaration_yaml,dict):
            self.parameters=declaration_yaml["parameters"] if "parameters" in declaration_yaml else []
            self.inputs = declaration_yaml["inputs"] if "inputs" in declaration_yaml else []
            self.outputs = declaration_yaml["outputs"] if "outputs" in declaration_yaml else None
            self.body = declaration_yaml["body"] if "body" in declaration_yaml else []
            self.withArgs = declaration_yaml["with"] if "with" in declaration_yaml else {}
        else:
            self.parameters=[]
            self.body=declaration_yaml
            self.outputs=None
            self.inputs=[]
            self.withArgs ={}

    def instantiate(self, dMap, parameters=None):
        if parameters is None:
            parameters={}
        if "args" in parameters:
            parameters=parameters["args"]
        if isinstance(parameters,list):
            pMap={}
            for p in range(len(self.parameters)):
                pMap[self.parameters[p]]=parameters[p]
            parameters=pMap
        l=Layers(self.body,dMap,parameters,self.outputs,self.inputs,self.withArgs)

        return l


class Declarations:

    def __init__(self,declarations_yaml):
        self.declarationMap:{str:Declaration}={ x:Declaration(declarations_yaml[x]) for x in declarations_yaml}
        pass

    def __contains__(self, item):
        return item in self.declarationMap

    def __getitem__(self, item):
        return self.declarationMap[item]

    def instantiate(self,name,inputs):
        v=self[name].instantiate(self)

        inp=self[name].inputs
        if len(inp)>0:
            if isinstance(inputs,list):
                pMap = {}
                for p in range(len(inp)):
                    pMap[inp[p]] = inputs[p]
                inputs=pMap
                pass
            if isinstance(inputs, dict):
                pass
        else:
            inputs={"$input":inputs}
        return v.build(inputs)

    def model(self,name,inputs):
        m=self.instantiate(name,inputs)
        return keras.Model(inputs,m)


import yaml

def load(path):
    with open(path, "r") as f:
        return yaml.load(f);

def create_model(path,inputs,name="net")->keras.Model:

    n=load(path)
    d=Declarations(n["declarations"])

    input_ = inputs
    out=d.model(name, inputs)
    return out

