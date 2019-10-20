from musket_core import coders,datasets
from musket_core.datasets import PredictionItem
from musket_core import context


class GenericCSVDataSet(datasets.DataSet):
    

    def processGroups(self, ctypes, input_groups, consumedInputs, rs):
        for i in input_groups:
            clns = input_groups[i]
            vls = []
            for q in clns:
                consumedInputs.add(q)
                v = self.data[q].values
                v = self.transformType(v, ctypes[q])
                vls.append(v)
            
            r = coders.ConcatCoder(vls)
            rs.append(r)
                

    def __init__(self,path,input_columns,output_columns,image_path=[],ctypes={},input_groups={},output_groups={}):
        super().__init__()
        self.data=context.csv_from_data(path)
        self.inputs=[]
        self.outputs=[]
        self.ctypes=ctypes
        
        consumedInputs=set()
        consumedOutputs=set()
        self.processGroups(ctypes, input_groups, consumedInputs, self.inputs)    
        self.processGroups(ctypes, output_groups, consumedOutputs, self.outputs)
            
               
        for i in input_columns:
            if i in consumedInputs:
                continue
            v=self.data[i].values;
            if i in ctypes:
                v=self.transformType(v,ctypes[i])
            self.inputs.append(v)
            
                
        for i in output_columns:
            if i in consumedOutputs:
                continue
            v=self.data[i].values;
            if i in ctypes:
                v=self.transformType(v,ctypes[i])
            self.outputs.append(v)
            
    def transformType(self,values,tpe):
        if tpe=="class":
            return coders.ClassCoder(values)
        if tpe=="number":
            return coders.NumCoder(values)
        if tpe=="str":
            return values
        return values        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item)->datasets.PredictionItem:
        inputs=[i[item] for i in self.inputs]
        outputs=[i[item] for i in self.outputs]
        return PredictionItem(item,inputs,outputs)
