import importlib
import os
import yaml
class Module:

    def __init__(self,dict):
        self.catalog={};
        self.entry=None
        for v in dict["types"]:
            t=Type(self, dict["types"][v]);
            if t.entry:
                self.entry=v;
            self.catalog[v.lower()]=t
            self.catalog[v] = t
        self.pythonModule=importlib.import_module(dict["(meta.module)"])
        pass

    def instantiate(self,dct,clearCustom=False,withArgs={}):
        if self.entry:
            typeDefinition = self.catalog[self.entry];
            clazz = getattr(self.pythonModule, self.entry)
            args = typeDefinition.constructArgs(dct,clearCustom)
            return clazz(**args)

        if type(dct)==dict:
            result = [];


            for v in dct:
                if hasattr(self.pythonModule,v[0].upper()+v[1:]):
                    clazz = getattr(self.pythonModule, v[0].upper()+v[1:])
                else: clazz=getattr(self.pythonModule, v)


                typeDefinition=self.catalog[v.lower()];
                args=typeDefinition.constructArgs(dct[v],clearCustom)
                allProps=typeDefinition.all()
                for v in withArgs:
                    if v in allProps:
                        args[v]=withArgs[v]
                if type(args)==dict:
                    result.append(clazz(**args))
                else:
                    result.append(clazz(args))
            return result


        return dct



class Type:

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
                    argMap[pos[i]]=dct[i]
                dct=argMap
                pass
            if isinstance(dct, dict):
                r=dct.copy()
                ct=self.custom()
                for v in dct:
                    if v in ct:
                        del r[v]
                return r
        return dct

    def alias(self,name:str):
        if name in self.properties:
            p:Property=self.properties[name]
            if p.alias!=None:
                return p.alias
        return name

    def custom(self):
        c= {v for v in self.properties if self.properties[v].custom}
        if self.type.lower() in self.module.catalog:
            c = c.union(self.module.catalog[self.type.lower()].custom())
        return c

    def positional(self):
        c= [v for v in self.properties if self.properties[v].positional]
        if self.type.lower() in self.module.catalog:
            c = c+self.module.catalog[self.type.lower()].positional()
        return c

    def all(self):
        c= [v for v in self.properties]
        if self.type.lower() in self.module.catalog:
            c = c+self.module.catalog[self.type.lower()].all()
        return c

    def __init__(self,m:Module,dict):
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
                    self.properties[p]=Property(self,dict['properties'][pOrig])
        else:
            self.type = dict
        pass

class Property:
    def __init__(self,t:Type,dict):
        self.type=t;
        self.alias=None
        self.positional="(meta.positional)" in dict
        self.custom = "(meta.custom)" in dict
        if "(meta.alias)" in dict:
            self.alias=dict["(meta.alias)"]
        pass

loaded={}

def load(name: str)  -> Module:
    if name in loaded:
        return loaded[name]
    pth = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pth,"schemas", name+".raml"), "r") as f:
        cfg = yaml.load(f);
    loaded[name]=Module(cfg);
    return loaded[name]

def parse(name:str,p):
    m=load(name)
    if type(p)==str:
        with open(p, "r") as f:
            return m.instantiate(yaml.load(f));
    return m.instantiate(p);