from musket_core import visualization,datasets
from musket_core import dataset_analis
from musket_core import losses
import  numpy as np
import operator
import string
from tqdm import tqdm
from sklearn import metrics
from numpy import ndarray

def checker(origin=None): 
    def inner(func): 
        func.checker=origin
        return func        
    return inner #this is the fun_obj mentioned in the above content


@visualization.dataset_analizer
def dataset_balance(y):
    s=(y>0.5).sum()
    if s>0:
        return "Positive samples"
    return "Negative samples"

def segmentationChecker(dataset):
    shp=dataset[0].y
    if isinstance(shp,ndarray):
        return len(shp.shape)==2 or len(shp.shape)==3
    return False

@visualization.dataset_analizer
@checker(segmentationChecker)
def positive_area(i,p:datasets.PredictionItem):
    y=p.y
    s=(y>0.5).sum()
    if s==0:
        return "0"
    alli=(y>=0.0).sum()
    if all==s:
        return "1"
    vl=int((s*10)/alli)
    return str(float(vl)/10)+"-"+str(float(vl+1)/10)



@visualization.prediction_analizer
@checker(segmentationChecker)
def iou_analizer(i,p:datasets.PredictionItem,pr:datasets.PredictionItem):
    vl=losses.iou_numpy(pr.prediction>0.5, pr.y, 0.0000001, 1)
    if vl==0:
        return "0"
    if vl==1:
        if pr.y.sum()==0:
            return "True negative"
        return "1"
    vl=int(vl*10)
    return str(float(vl)/10)+"-"+str(float(vl+1)/10)

def classificationChecker(dataset):
    shp=dataset[0].y
    if isinstance(shp,ndarray):
        return len(shp.shape)==1
    return False

@visualization.dataset_analizer
@checker(classificationChecker)
def onehot_analizer(y):
    r=np.where(y>0.5)
    if len(r[0])>0:
        return int(r[0][0])
    return "Negative sample"


@visualization.dataset_analizer
@checker(classificationChecker)
def class_num_analizer(y):
    if isinstance(y, np.ndarray):
        return int(y[0])
    return int(y)


class FixedShape:
    def __init__(self,shape,pos):
        self.shape=shape
        self.pos=pos

    def range(self):
        return self.shape

class SimpleShapeRange:
    def __init__(self,rs,pos):
        self.shape=rs
        self.pos=pos
    def range(self):
        if isinstance(self.shape,dict):
            arr=np.array(list(self.shape.keys()));
            minv=arr.min()
            maxv=arr.max()
            return slice(minv,maxv,1)
        mv=[]
        for x in self.shape:
            mv.append(x.range())
        return mv

    def split(self,shapes,pos):
        if isinstance(self.shape, dict):
            return self.inner_split(shapes,pos)
        rs=[]
        for x in self.shape:
            if isinstance(x,FixedShape):
                continue
            d = x.split(shapes, pos + [self.pos])
            if isinstance(d, list):
                rs=rs+d
            else: rs.append(d)
        return rs

    def inner_split(self,shapes,pos):
        hist={}
        
        allBins=sorted(list(shapes.keys()))
        totalCount=0
        for x in allBins:
            totalCount=totalCount+len(shapes[x])
        nb=totalCount/10
        hist={}
        cb=[]
        all=[]
        allRanges=[]
        lastStart=None
        for s in allBins:
            cw=s
            for x in pos:
                s=s[x]
            if lastStart is None:
                lastStart=s    
            cw_ = shapes[cw]
            cb=cb+cw_
            hist[s]=len(cw_)
            if len(cb)>nb:
               all.append(cb)
               allRanges.append((lastStart,s)) 
               lastStart=s
               cb=[]
        if len(cb)>0:       
            all.append(cb)
            allRanges.append((lastStart,s)) 
            lastStart=s
            cb=[]
        res={}    
        for x in range(len(allRanges)):
            res[allRanges[x]]=all[x]                
        return (hist,res)    
        


def _process_shapes(shapes):
        shapesInPos = {}

        for sh in shapes:
            if isinstance(sh,int):
                return shapes
            dim = len(sh)
            for i in range(dim):
                shapeInPos = sh[i]

                if i in shapesInPos:
                    local_dict = shapesInPos[i]
                else:
                    local_dict = {}
                    shapesInPos[i] = local_dict
                if shapeInPos in local_dict:
                    local_dict[shapeInPos] = local_dict[shapeInPos] + 1
                else:
                    local_dict[shapeInPos] = 1
        res=[]
        for pos in shapesInPos:
            dict=shapesInPos[pos]
            if len(dict)==1:
                for k in dict.keys():
                    res.append(FixedShape(k, pos))
            else:
                rs=_process_shapes(dict)
                res.append(SimpleShapeRange(rs,pos))
        return res


class InputShapeAnalizer:

    def __init__(self):
        self.shapes={}
        self._build=None
        pass

    def __call__(self, index,p:datasets.PredictionItem):
        sh=dataset_analis.get_shape(p.x)
        if sh in self.shapes:
            self.shapes[sh].append(index)
        else:
            self.shapes[sh]=[index]
        
        pass

    def build(self):

        shapesInPos = _process_shapes(self.shapes)
        res=[]
        for i in range(len(shapesInPos)):
            sh=shapesInPos[i]
            if isinstance(sh,SimpleShapeRange):
                res.append(sh.split(self.shapes,[i]))
                
        self._build=res        
        pass


    def results(self):
        if self._build is None:
            self.build()
            
        r= [x[1] for x in self._build]
        if len(r)>0:
            return r[0]
        return {}

    def visualizationHints(self):
        r= [x[0] for x in self._build]
        return { "type":"hist","values":r}

@visualization.dataset_analizer
class LengthAnalizer:

    def __init__(self):
        self.shapes={}
        self.looksLikeBinary=True
        self.all=InputShapeAnalizer()
        self.positive=InputShapeAnalizer()
        self.negative=InputShapeAnalizer()

    def __call__(self, index, p: datasets.PredictionItem):
        self.all(index,p)
        if self.looksLikeBinary:
            if isinstance(p.y,np.ndarray):
                if len(p.y.shape)==1 and p.y.shape[0]==1:
                    if p.y[0]==0:
                        self.negative(index,p)
                    if p.y[0]==1:
                        self.positive(index,p)
                    return
        self.looksLikeBinary=False
        pass

    def build(self):
        self.all.build()
        if self.looksLikeBinary:
            self.positive.build()
            self.negative.build()
        pass

    def results(self):
        if not self.all._build:
            self.build()
        return self.all.results()

    def visualizationHints(self):
        if not self.all._build:
            self.build()
        if not self.looksLikeBinary:
            return self.all.visualizationHints()
        r = [x[0] for x in self.all._build]
        rp = [x[0] for x in self.positive._build]
        rn = [x[0] for x in self.negative._build]
        if len(r)>0:
            res=[r[0]]
            if len(rp)>0 and len(rn)>0:
                res= [r[0],rp[0],rn[0]]
            return {"type": "hist", "values": res}
        return None

class MultiClassMetricsAnalizer:
    def __init__(self):
        self.shapes={}
        self.looksLikeBinary=True
        self.predictions=[]
        self.ground_truth=[]
        self.ids=[]
        self.gt=[]
    
    def __call__(self, index, p: datasets.PredictionItem,prediction:datasets.PredictionItem,**args):   
        x=np.where(prediction.prediction==prediction.prediction.max())
        #prediction.prediction=np.zeros(prediction.prediction.shape)
        #prediction.prediction[x[0]]=1
        self.predictions.append(prediction.prediction)     
        self.ground_truth.append(prediction.y)
        self.ids.append(prediction.id)
        pass
    
    def visualizationHints(self):
        return {"type": "hist", "values": self.scores,"x_axis":"Class","y_axis":"F1 Score"}    

@visualization.prediction_analizer
@checker(classificationChecker)
class MultiClassF1Analizer(MultiClassMetricsAnalizer):
    

    def results(self):
        preds=np.array(self.predictions)
        gt=np.array(self.ground_truth)
        scores=[]
        if len(preds.shape)>2:
            preds=np.concatenate(self.predictions,axis=0)
            gt=np.concatenate(self.ground_truth,axis=0)
            pass
        for i in range(preds.shape[1]):
            pri=preds[:,i]
            gti=gt[:,i]
            f1=metrics.f1_score(gti,pri>0.5)
            scores.append(float(f1))
        self.scores=scores;
        return []
    
    
@visualization.prediction_analizer
@checker(classificationChecker)
class MultiClassPrecisionAnalizer(MultiClassMetricsAnalizer):
    

    def results(self):
        preds=np.array(self.predictions)
        gt=np.array(self.ground_truth)
        if len(preds.shape)>2:
            preds=np.concatenate(self.predictions,axis=0)
            gt=np.concatenate(self.ground_truth,axis=0)
            pass
        scores=[]
        for i in range(preds.shape[1]):
            pri=preds[:,i]
            gti=gt[:,i]
            f1=metrics.precision_score(gti,pri>0.5)
            scores.append(float(f1))
        self.scores=scores;
        return []    
    
    def visualizationHints(self):
        return {"type": "hist", "values": self.scores,"x_axis":"Class","y_axis":"Precision"}    

@visualization.prediction_analizer
@checker(classificationChecker)
class MultiClassRecallAnalizer(MultiClassMetricsAnalizer):
    

    def results(self):
        preds=np.array(self.predictions)
        gt=np.array(self.ground_truth)
        if len(preds.shape)>2:
            preds=np.concatenate(self.predictions,axis=0)
            gt=np.concatenate(self.ground_truth,axis=0)
            pass
        scores=[]
        for i in range(preds.shape[1]):
            pri=preds[:,i]
            gti=gt[:,i]
            f1=metrics.recall_score(gti,pri>0.5)
            scores.append(float(f1))
        self.scores=scores;
        return []    
    
    def visualizationHints(self):
        return {"type": "hist", "values": self.scores,"x_axis":"Class","y_axis":"Recall"}
    
@visualization.dataset_analizer
@checker(classificationChecker)
class MultiClassFrequenceyAnalizer:

    def __init__(self):
        self.shapes={}
        self.looksLikeBinary=True
        self.freq={}
        self.count=0
        

    def __call__(self, index, p: datasets.PredictionItem=None,prediction:datasets.PredictionItem=None,**args):
        self.count=self.count+1
        for  v in np.where(p.y>0.5)[0]:
            if v in self.freq:
                self.freq[v].append(index)
            else: self.freq[v]=[p.id]      
        pass

    

    def results(self):
        m=max(self.freq.keys())
        scores=np.zeros(m)
        
        for v in range(m):
            if v in self.freq:
                scores[v]=len(self.freq[v])
            else:
                scores[v]=0    
        self.scores=[float(x)/self.count for x in scores];
        return self.freq

    def visualizationHints(self):
        return {"type": "hist", "values": self.scores,"x_axis":"Class","y_axis":"Frequency"}    
        

@visualization.prediction_analizer
@checker(classificationChecker)
def ground_truth_vs_prediction(x,y):
    
    allCorrect=np.equal(x>0.5,y>0.5).sum()==len(x)
    if allCorrect:
        return "Correct"
    return "Incorrect"

@visualization.prediction_analizer
@checker(classificationChecker)
class MultiOutputCategoricalAnalizer(MultiClassMetricsAnalizer):
    

    def results(self):
        preds=np.array(self.predictions)
        gt=np.array(self.ground_truth)
        scores=[]
        
        for i in range(preds.shape[1]):
            pri=preds[:,i]
            gti=gt[:,i]
            
            pri=np.argmax(pri,axis=1)
            gti=np.argmax(gti,axis=1)
            
            f1=metrics.accuracy_score(gti,pri)
            scores.append(float(f1))
        self.scores=scores;
        return []    
    
    def visualizationHints(self):
        return {"type": "hist", "values": self.scores,"x_axis":"Output","y_axis":"Categorical Accuracy"}
    

@visualization.prediction_analizer
def void_analizer(x,y):
    return "Ok"

@visualization.dataset_analizer
def void_data_analizer(*args):
    return "Ok"

class WordDS(datasets.DataSet):
    
    def __init__(self,sorted):
        self.sorted=sorted
        
    
    def __len__(self):
        return len(self.sorted)
    
    
    def __getitem__(self, item)->datasets.PredictionItem:
        return datasets.PredictionItem(self.sorted[item][0],self.sorted[item][0],self.sorted[item][1])

class WordDS2(datasets.DataSet):
    
    def __init__(self,sorted,l):
        self.sorted=sorted
        self._len=l
        
    
    def __len__(self):
        return self._len
    
    
    def __getitem__(self, item)->datasets.PredictionItem:
        return datasets.PredictionItem(self.sorted[item][0],self.sorted[item][0],(self.sorted[item][1],"count:"+str(self.sorted[item][2])))        

def textDS(dataset):
    for i in range(len(dataset)):
        shp=dataset[i].x
        if len(shp)>0:
            return isinstance(shp[0],str)
    return False

@visualization.dataset_analizer
@checker(textDS)
class VocabularyAnalizer:
    def __init__(self):
        self.components={}
        self.words= {}
        self.doccount=0
        pass

    def __call__(self, index, p: datasets.PredictionItem):
        self.doccount=self.doccount+1
        for a in p.x:
            if a in self.components:
                self.components[a].add(index)
                self.words[a]=self.words[a]+1
            else:
                self.components[a] = {index}
                self.words[a]=1
        pass

    def results(self):
        sorted_x = sorted(self.words.items(), key=operator.itemgetter(1))
        return  { "vocabulary": WordDS(sorted_x)}

    def visualizationHints(self):
        sorted_x = sorted(self.words.items(), key=operator.itemgetter(1))

        mn:set=set()
        
        res={}
        count=0
        for i in tqdm(range(len(sorted_x))):
            x=sorted_x[i]
            word=x[0]
            mn|=self.components[word]
            count=count+1
            res[len(self.words)-count]=len(mn)
        return {"type": "hist", "values": [res],"total":self.doccount,"x_axis":"Size of vocubalary","y_axis":"Fraction of fully covered documents"}
    
@visualization.dataset_analizer
@checker(textDS)
class WordFrequencyAnalizer:
    def __init__(self,min_count:int=5,exclude_stop_words:bool=False,stopWordsLanguage:str="english",exclude_punctuation:bool=False,top_n:int=20):
        self.analizers={}
        self.min_count=min_count
        self.stopwords=exclude_stop_words
        self.analizers["all"]=VocabularyAnalizer()
        self.exclude_punctuation=exclude_punctuation
        self.top_n=top_n
        if self.stopwords:
            try:
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words(stopWordsLanguage))
            except:
                import nltk
                nltk.download('stopwords')
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words(stopWordsLanguage))
        else:
            self._stopwords=None
        pass

    def __call__(self, index, p: datasets.PredictionItem):
        
        self.analizers["all"](index,p)
        ids=np.where(p.y>0)
        for k in ids[0]:
            m=int(k)
            if m not in self.analizers:
                self.analizers[m]=VocabularyAnalizer()
            self.analizers[m](index,p)
        pass

    def results(self):
        allAnalizer=self.analizers["all"]
        res={}
        for v in self.analizers:
            if v!="all":
               analizer=self.analizers[v]
               rs=[]
               for x,y in analizer.words.items():
                   if allAnalizer.words[x]>self.min_count:
                       rs.append((x,y/allAnalizer.words[x],allAnalizer.words[x]))
               sorted_x = list(reversed(sorted(rs, key=operator.itemgetter(1))))
               res[v]=WordDS2(sorted_x,len(rs))
        return res

    def visualizationHints(self):
        allAnalizer = self.analizers["all"]
        sorted_x = reversed(sorted(allAnalizer.words.items(), key=operator.itemgetter(1)))
        res={}
        for i in sorted_x:
            if self.exclude_punctuation and i[0] in string.punctuation:
                continue
            if self._stopwords is not None:
                if i[0] in self._stopwords or i[0].lower() in self._stopwords:
                    continue
            res[i[0]]=i[1]

            if len(res)>self.top_n:
                break
        return {"type": "bar", "values": [res], "total": allAnalizer.doccount, "x_axis": "Most Frequent words",
                     "y_axis": "Usage Count"}



@visualization.dataset_analizer
@checker(classificationChecker)
def multi_class_balance(y):
    if isinstance(y, np.ndarray):
        return int(y[0])
    return int(y)

@visualization.dataset_filter
def contains_string(y:datasets.PredictionItem,filter:str):
    val=y.x
    return filter in str(val)

@visualization.dataset_filter
def has_class(y:datasets.PredictionItem,filter:str):
    val=y.y
    if len(val)>1:
        return val[int(filter)]==1
    return val[0]==int(filter)

@visualization.dataset_filter
def custom_python(y:datasets.PredictionItem,filter:str):
    val=y.x
    return filter in str(val)
