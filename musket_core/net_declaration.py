from musket_core import configloader
import keras
import musket_core.templating as tp
layers=configloader.load("layers")

import inspect

class Layers:

    def __init__(self,layers_yaml,declarations,parameters,outputs=None,linputs=None):
        layers_yaml=tp.resolveTemplates(layers_yaml,parameters)
        pName = "$input"
        self.layerMap:{str: keras.layers.Layer}={}
        self.layerInputs:{str:[str]} = {}
        self.layerArguments:{str:{str}}={}
        self.layerSequence:[keras.layers.Layer]=[]
        self.name=""
        nums={}
        for layer in layers_yaml:
            layerImpl=None
            key=list(layer.keys())[0]
            config = layer[key]
            if key in declarations:
                decl=declarations[key]
                layerImpl=decl.instantiate(declarations,config)
                inputs = config["inputs"] if "inputs" in config else pName
                if key in nums:
                    num=nums[key]
                    nums[key]=nums[key]+1
                else:
                    num=0
                    nums[key]=1
                if "name" in config:
                    name=config["name"]
                else: name=key+str(num)
                layerImpl.name=name
            else:
                layerImpl = layers.instantiate(layer, True)[0]
                name = layer["name"] if "name" in layer else layerImpl.name
                if isinstance(config,dict):
                    inputs = config["inputs"] if "inputs" in config else pName
                else:
                    inputs=pName

            if inputs==pName:
                if isinstance(config,list):
                    all_refs=True
                    for v in config:
                        if v in self.layerMap or v in linputs:
                            pass
                        else:
                            all_refs=False
                    if all_refs:
                        inputs=config
                    pass
            self.layerMap[name]=layerImpl
            self.layerInputs[name]=inputs
            self.layerArguments[name]=config
            self.layerSequence.append(layerImpl)
            pName=name
            self.output=name
            if outputs is not None:
                self.output=outputs
        pass

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
        else:
            self.parameters=[]
            self.body=declaration_yaml
            self.outputs=None
            self.inputs=[]

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
        l=Layers(self.body,dMap,parameters,self.outputs,self.inputs)

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

