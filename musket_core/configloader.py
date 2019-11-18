import importlib
import os
import yaml
import inspect
import keras
from musket_core import utils

builtIns = {
    "any": True,
    "object": True,
    "array": True,
    "string": True,
    "number": True,
    "integer": True,
    "date": True,
    "boolean": True
}

def isBuiltIn(typeName:str)->bool:
    return typeName in builtIns

def get_object(name):
    vs=load("layers").catalog[name]
    return vs 


def register(module):
    load("layers").register_module(module)

def has_object(name):
    return name in load("layers").catalog

class Module:

    def __init__(self,dict):
        self.catalog={};
        self.orig={}
        self.dependencies={}
        self.entry=None
        for v in dict["types"]:
            t=Type(v,self,dict["types"][v]);
            if t.entry:
                self.entry=v;
            self.catalog[v.lower()]=t
            self.catalog[v] = t
            self.orig[v.lower()]=v
        self.pythonModule = None
        if "(meta.module)" in dict:
            self.pythonModule=importlib.import_module(dict["(meta.module)"])
        self.register_module(self.pythonModule,False)
        pass

    def register(self,pythonPath):
        pyMod=importlib.import_module(pythonPath)
        self.register_module(pyMod)

    def register_module(self, pyMod,override=True):
        for x in dir(pyMod):
            if not override:
                if x.lower() in self.catalog:
                    continue
            tp = getattr(pyMod, x)
            if inspect.isclass(tp):
                if x in self.catalog:
                    tp = self.catalog[x]
                else:
                    init = getattr(tp, "__init__")
                    try:
                        gignature = inspect.signature(init)
                        typeName = x
                        tp = PythonType(typeName, gignature, tp)
                        self.catalog[typeName.lower()] = tp
                        self.catalog[typeName] = tp
                        self.orig[typeName.lower()] = typeName
                    except:
                        e: ValueError

            if inspect.isfunction(tp):
                if x.lower() in self.catalog:
                    continue
                gignature = inspect.signature(tp)
                typeName = x
                tp = PythonFunction(gignature, tp)
                self.catalog[typeName.lower()] = tp
                self.catalog[typeName] = tp
                self.orig[typeName.lower()] = typeName
            pass
        pass

    def instantiate(self, dct, clear_custom=False, with_args=None):
        if with_args is None:
            with_args={}
        if self.entry:
            typeDefinition = self.catalog[self.entry];
            clazz = getattr(self.pythonModule, self.entry)
            args = typeDefinition.constructArgs(dct, clear_custom)
            return clazz(**args)

        if type(dct)==dict:
            result = []

            for v in dct:
                typeDefinition = self.catalog[v.lower()]
                if isinstance(typeDefinition,PythonType) or isinstance(typeDefinition,PythonFunction):
                    clazz=typeDefinition.clazz
                else:
                    if hasattr(self.pythonModule,v[0].upper()+v[1:]):
                        clazz = getattr(self.pythonModule, v[0].upper()+v[1:])
                    else: clazz=getattr(self.pythonModule, self.orig[v])

                args=typeDefinition.constructArgs(dct[v], clear_custom)
                allProps=typeDefinition.allProperties()
                for v in with_args:
                    if v in allProps:
                        args[v]=with_args[v]
                if type(args)==dict:
                    result.append(clazz(**args))
                else:
                    result.append(clazz(args))
            return result


        return dct

    def addDependency(self, key:str, m):
        self.dependencies[key] = m

    def getType(self, name:str):
        if isBuiltIn(name):
            return None
        module = self
        shortName = name
        if "." in name:
            ind = name.index(".")
            namespace = name[:ind]
            if not namespace in self.dependencies:
                raise ValueError(f"Unknown namespace '{namespace}'")
            module = module.dependencies[namespace]
            shortName = name[ind+1:]

        if not shortName in module.catalog:
            raise ValueError(f"Can not resolve type '{name}'")

        result = module.catalog[shortName]
        return result

class AbstractType:

    def __init__(self):
        self.name = None
        self.type = None
        self.module=None

    def positional(self):
        return []

    def allProperties(self):
        return []

    def customProperties(self):
        return []

    def property(self, propName): None

    def constructArgs(self,dct,clearCustom=False):
        #for c in dct:
        
        if type(dct)==str or type(dct)==int:

            return dct

        if clearCustom:

            if isinstance(dct,dict) and "args" in dct:
                dct=dct["args"]
            if isinstance(dct,list):
                pos=self.positional()
                argMap={}
                for i in range(min(len(dct),len(pos))):
                    prop = pos[i]
                    if isinstance(prop,str):
                        try:
                            v=self.property(prop)
                            if v is not None:
                                prop=v
                        except:
                            pass          
                    value = dct[i]
                    if self.module != None  and isinstance(prop,Property) and prop.propRange in self.module.catalog:
                        propRange = self.module.catalog[prop.propRange]
                        if propRange.isAssignableFrom("Layer"):
                            value = self.module.instantiate(value, True,{})[0]
                    if isinstance(prop,str):
                        argMap[prop]=value
                    else: argMap[prop.name]=value
                dct=argMap
                pass
            if isinstance(dct, dict):
                r=dct.copy()
                ct=self.customProperties()
                for v in dct:
                    if v in ct:
                        del r[v]
                return r
        return dct

    def isAssignableFrom(self, typeName):
        if self.name == typeName:
            return True
        elif self.type == None:
            return False
        elif self.type == typeName:
            return True
        else:
            for t in self.superTypes():
                if t.isAssignableFrom(typeName):
                    return True
            return False

    def superTypes(self):
        names = self.type if isinstance(self.type, list) else [ self.type ]
        result = []
        for n in names:
            if isinstance(n, str):
                st = self.module.getType(n)
                if st is not None:
                    result.append(st)
        return result



class PythonType(AbstractType):

    def __init__(self,type:str,s:inspect.Signature,clazz):
        super(PythonType, self).__init__()
        args=[p for p in s.parameters][1:]
        self.args=args
        self.clazz=clazz
        self.type = type

    def positional(self):
        return self.args

    def allProperties(self):
        return self.args

class PythonFunction(AbstractType):

    def __init__(self,s:inspect.Signature,clazz):
        super(PythonFunction,self).__init__()
        # if clazz.__name__.lower()=="flatten":
        #     print("A")
        self.func=clazz
        if hasattr(clazz,"args"):
            args=[p for p in clazz.args if "input" not in p and "inp" not in p]
        else: args=[p for p in s.parameters if "input" not in p and "inp" not in p]
        self.args=args
        inpP="input"
        if "inp" in s.parameters:
            inpP="inp"
           
        def create(*args,**kwargs):
            if len(args) == 1 and args[0] is None:
                args = []
            mm = kwargs.copy()
            for i in range(len(args)):
                mm[self.args[i]]=args[i]
            def res(i):
                if isinstance(i, list) or isinstance(i , tuple):
                    if len(i)==0:
                        i=None
                if i is not None:
                    mm[inpP]=i

                try:
                    result=clazz(**mm)
                except:
                    import traceback
                    traceback.print_exc()
                    result = None
                
                if result is None:
                    print(f"{clazz} returned None")    

                return result
            return res
        self.clazz=create

    def positional(self):
        return self.args

    def allProperties(self):
        return self.args

class Type(AbstractType):


    def __init__(self,name:str,m:Module,dict):
        super(Type,self).__init__()
        self.name = name
        self.module=m;
        self.properties={};
        self.entry="(meta.entry)" in dict
        if type(dict)!=str:
            self.type=dict["type"]
            if 'properties' in dict:
                for p in dict['properties']:
                    pOrig=p;
                    if p[-1]=='?':
                        p=p[:-1]
                    self.properties[p]=Property(p,self,dict['properties'][pOrig])
        else:
            self.type = dict
        pass

    def alias(self,name:str):
        if name in self.properties:
            p:Property=self.properties[name]
            if p.hasAnnotation("alias"):
                return p.annotation("alias")
        for st in self.superTypes():
            als = st.alias(name)
            if als != name:
                return als
        return name

    def customProperties(self):
        result = self.gatherBooleanAnnotatedProperties("custom")
        return result

    def property(self, propName):
        if propName == None:
            return None
        if propName in self.properties:
            return self.properties[propName]
        elif self.type.lower() in self.module.catalog:
            return self.module.catalog[self.type.lower()].property(propName)
        else:
            return None

    def positional(self):
        result = self.gatherBooleanAnnotatedProperties("positional")
        return result

    def allProperties(self):
        c= [v for v in self.properties]
        for st in self.superTypes():
            c += st.allProperties()
        return c

    def gatherAnnotatedProperties(self, aName:str):
        result = {}
        for st in self.superTypes():
            stProps = st.gatherAnnotatedProperties(aName)
            for pName in stProps:
                result[pName] = stProps[pName]
        c = [self.properties[v] for v in self.properties if (self.properties[v].hasAnnotation(aName))]
        for prop in c:
            result[prop.name] = prop
        return result

    def gatherBooleanAnnotatedProperties(self, aName:str):
        result = self.gatherAnnotatedProperties(aName)
        keys = [x for x in result]
        for pName in keys:
            aValue = result[pName].annotation(aName)
            if isinstance(aValue,bool) and aValue:
                pass
            else:
                result.pop(pName,None)
        return [x for x in result]


class Property:
    def __init__(self,name:str,t:Type,dict1):
        self.name=name
        self.type=t;

        self.annotations = {}
        if isinstance(dict1,dict):
            for annotationName in ["positional", "custom", "alias"]:
              key = f"(meta.{annotationName})"
              if key in dict1:
                  self.annotations[annotationName]= dict1[key]

            if "type" in dict1:
                self.propRange = dict1["type"]

        elif isinstance(dict1, str):
            self.propRange = dict1
        else:
            self.propRange = "string"
        pass

    def hasAnnotation(self, aName: str):
        return aName in self.annotations

    def annotation(self,aName:str):
        if aName in self.annotations:
            return self.annotations[aName]
        return None


loaded={}

def yaml_load(f):
    return yaml.load(f, Loader=yaml.Loader)

def load(name: str)  -> Module:
    if name in loaded:
        return loaded[name]
    pth = os.path.dirname(os.path.abspath(__file__))
    fName = name if name.endswith('.raml') else name + ".raml"
    with open(os.path.join(pth,"schemas", fName), "r") as f:
        cfg = yaml_load(f);
    result = Module(cfg)
    loaded[name]= result;
    if 'uses' in cfg:
        uses = cfg['uses']
        for key in uses:
            mPath = uses[key]
            m = load(mPath)
            result.addDependency(key,m)

    return result

alllowReplace=["declarations","callbacks","datasets","tasks"]
def parse(name:str,p,extra=None):
    m=load(name)

    if type(p)==str:
        with open(p,"r") as f:
            first_line = f.readline()
            first_line=first_line.strip()
            if first_line.startswith("#%Musket "):
                first_line=first_line[8:].strip()
                dialect=first_line[:first_line.index(' ')].strip()
                m=load(dialect.lower())
                #dialect=
        with open(p, "r") as f:
            base=yaml_load(f)

            if extra is not None:
                extrad=utils.load_yaml(extra)
                for v in extrad:
                    if v not in base:
                        base[v]=extrad[v]
                    else:
                        if v in alllowReplace:
                            for q in extrad[v]:
                                mn=base[v]
                                if mn is not None:
                                    if q not in mn:
                                        mn[q]=extrad[v][q]
                                else:
                                    base[v]=extrad[v]
                                    break        

            return m.instantiate(base)
    return m.instantiate(p)
