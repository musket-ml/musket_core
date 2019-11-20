from musket_core import coders,datasets,utils
from musket_core.datasets import PredictionItem
from musket_core import context
import pandas as pd
import os
class GenericCSVDataSet(datasets.DataSet):
    

    def processGroups(self, ctypes, input_groups, consumedInputs, rs):
        for i in input_groups:
            clns = input_groups[i]
            vls = []
            for q in clns:
                consumedInputs.add(q)
                v = self.data[q].values
                v = self.transformType(q,v, ctypes[q])
                vls.append(v)
            
            r = coders.ConcatCoder(vls)
            rs.append(r)
            
    def _encode_dataset(self,ds,encode_y=False,treshold=0.5):
        if len(self.output_groups)>0:
            raise NotImplementedError("Output groups support is not implemented yet")
        return super()._encode_dataset(ds, encode_y, treshold);
    
    def _create_dataframe(self,items):    
        return pd.DataFrame(items,columns=self.data.columns)
            
    def _encode_item(self,item:PredictionItem,encode_y=False,treshold=0.5):
        res={}        
        data=self.data.values[item.id]
        pr=item.prediction
        if len(self.outputs)==1:
            pr=[pr]
        for i in range(len(self.data.columns)):
            cln=self.data.columns[i]
            if cln in self.output_columns_set and not encode_y:
                nm=self.output_columns.index(cln)
                res[cln]=self.outputs[nm]._decode(pr[nm],treshold)                
            else:
                res[cln]=data[i]

        return res
              
    def init_coders_from_path(self,path):
        ph=os.path.join(path,"assets")
        for c in self.coders:
            coderPath=os.path.join(ph,c.replace("|","_")+".cm")
            if os.path.exists(coderPath):
                dt=utils.load(coderPath)
                self.coders[c].init_from_meta(dt)
                  

    def __init__(self,path,input_columns,output_columns,image_path=[],ctypes={},input_groups={},output_groups={}):
        super().__init__()
        self.data=context.csv_from_data(path)
        self.inputs=[]
        self.outputs=[]
        self.ctypes=ctypes
        self.coders={}
        self.output_columns=output_columns
        self.output_columns_set=set(output_columns)
        self.imagePath=image_path
        self.output_groups=output_groups
        for c in output_columns:
            if c not in self.data.columns:
                self.data.insert(len(self.data.columns),c,"")
        consumedInputs=set()
        consumedOutputs=set()
        self.imageCoders=[]
        self.processGroups(ctypes, input_groups, consumedInputs, self.inputs)    
        self.processGroups(ctypes, output_groups, consumedOutputs, self.outputs)
            
               
        for i in input_columns:
            if i in consumedInputs:
                continue
            v=self.data[i].values;
            if i in ctypes:
                v=self.transformType(i,v,ctypes[i])
            self.inputs.append(v)
            
                
        for i in output_columns:
            if i in consumedOutputs:
                continue
            else:    
                v=self.data[i].values;
            if i in ctypes:
                v=self.transformType(i,v,ctypes[i])
            self.outputs.append(v)
            
            
    def transformType(self,cln,values,tpe):
            if tpe=="as_is":
                return values
            cd= coders.get_coder(tpe, values,self)
            self.coders[cln]=cd
            if isinstance(cd, coders.ImageCoder):
                self.imageCoders.append(cd)
            return cd
                
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item)->datasets.PredictionItem:
        inputs=[i[item] for i in self.inputs]
        outputs=[i[item] for i in self.outputs]
        if isinstance(inputs, list) and len(inputs)==1:
            inputs=inputs[0]
        if isinstance(outputs, list) and len(outputs)==1:
            outputs=outputs[0]    
        return PredictionItem(item,inputs,outputs)
    
    