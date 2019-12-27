import unittest
import os
from musket_core import image_datasets
from _io import StringIO
import pandas as pd
import numpy as np
import math

if __name__ == '__main__':
    unittest.main()

fl=__file__

fl=os.path.dirname(fl)

def dummyImage(item):
    return np.zeros_like((200,200,3))
class TestCoders(unittest.TestCase):
    def test_binary_classification(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 0
        a2, 0
        a4, 1
        a3, 1
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.BinaryClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[0],True)
        
        encoded=ds.encode(ds, True)
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_binary_classification1(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 1
        a2,
        a4,
        a3,
            """)
        x=set({math.nan, 1.0,math.nan})
        r=math.nan in x
        z=x.remove(math.nan)
        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.BinaryClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],True)
        self.assertEqual(ds[2].y[0],False)
        
        #encoded=ds.encode(ds, True)
        #rs=encoded["Clazz"].values==df["Clazz"].values
        #self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_binary_classification2(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, y
        a2, n
        a4, n
        a3, n 
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.BinaryClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],True)
        self.assertEqual(ds[2].y[0],False)
        df['Clazz'] = df['Clazz'].str.strip()
        encoded=ds.encode(ds, True)
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_category_classification(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 0
        a2, 0
        a3, 1
        a4, 2
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.CategoryClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],True)
        self.assertEqual(ds[2].y[0],False)
        
        encoded=ds.encode(ds, True)
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_category_classification1(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, f
        a2, m
        a3, m
        a4, g
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.CategoryClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],True)
        self.assertEqual(ds[2].y[0],False)
        
        encoded=ds.encode(ds, True)
        df['Clazz'] = df['Clazz'].str.strip()
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_category_classification2(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 
        a2, m
        a3, m
        a4, g
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.CategoryClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],True)
        self.assertEqual(ds[2].y[0],False)
        
        encoded=ds.encode(ds, True)
        df['Clazz'] = df['Clazz'].str.strip()
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_category_classification3(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 
        a2, 0
        a3, 0
        a4, 1
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.CategoryClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],True)
        self.assertEqual(ds[2].y[0],False)
        
        encoded=ds.encode(ds, True)
        df['Clazz'] = df['Clazz'].str.strip()
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    
    def test_multi_class(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 
        a2, 0
        a3, 0
        a4, 1
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.MultiClassClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[0],True)
        
        encoded=ds.encode(ds, True)
        df['Clazz'] = df['Clazz'].str.strip()
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_multi_class1(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 
        a2, 0 1
        a3, 0
        a4, 1 2
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.MultiClassClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[0],True)
        
        encoded=ds.encode(ds, True)
        df['Clazz'] = df['Clazz'].str.strip()
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_multi_class2(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 
        a2, X|Y
        a3, X
        a4, X|Z
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.MultiClassClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[0],True)
        
        encoded=ds.encode(ds, True)
        df['Clazz'] = df['Clazz'].str.strip()
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_multi_class3(self):
        TESTDATA = StringIO("""Image,Clazz
        a1, 
        a2, X_Y
        a3, X
        a4, X_Z
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.MultiClassClassificationDataSet([],df,"Image","Clazz")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[0],True)
        
        encoded=ds.encode(ds, True)
        df['Clazz'] = df['Clazz'].str.strip()
        rs=encoded["Clazz"].values==df["Clazz"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_multi_class4(self):
        TESTDATA = StringIO("""Image,A,B
        a1, 0, 0
        a2, 1, 0
        a3, 0 ,1
        a4, 0 ,0
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.MultiClassClassificationDataSet([],df,"Image","A|B")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[1],True)
        
        encoded=ds.encode(ds, True)
        df['A'] = df['A']
        rs=encoded["A"].values==df["A"].values
        self.assertEqual(rs.sum(), 4, "")
        pass
    
    def test_multi_class5(self):
        TESTDATA = StringIO("""Image,A,B
        a1, n, n
        a2, y, n
        a3, n ,y
        a4, y ,n
            """)

        df = pd.read_csv(TESTDATA, sep=",") 
        ds=image_datasets.MultiClassClassificationDataSet([],df,"Image","A|B")
        ds.get_value=dummyImage
        
        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[1],True)
        
        encoded=ds.encode(ds, True)
        df['A'] = df['A']
        df['A'] = df['A'].str.strip()
        rs=encoded["A"].values==df["A"].values
        self.assertEqual(rs.sum(), 4, "")
        pass      
    
    
    def test_multi_class6(self):
        TESTDATA = StringIO("""Image,A,B
        a1, , 
        a2, y,
        a3,  ,y
        a4, y ,
            """)

        df = pd.read_csv(TESTDATA, sep=",")
        ds=image_datasets.MultiClassClassificationDataSet([],df,"Image","A|B")
        ds.get_value=dummyImage

        self.assertEqual(ds[0].y[0],False)
        self.assertEqual(ds[2].y[1],True)

        encoded=ds.encode(ds, True)
        df['A'] = df['A']
        df['A'] = df['A'].str.strip()
        rs=encoded["A"].values==df["A"].values
        self.assertEqual(rs.sum(), 4, "")
        pass

    def test_multi_class7(self):  #Test based on small piece of Severstal dataset
        TESTDATA = StringIO("""ImageId,ClassId
        ff6e35e0a.jpg,1
        ff6e35e0a.jpg,2
        ff933e271.jpg,3
        ff96dfa95.jpg,3
        ff9923932.jpg,3
        ff9d46e95.jpg,4
        ffb48ee43.jpg,3
        ffbd081d5.jpg,3
        ffc9fdf70.jpg,3
        ffcf72ecf.jpg,3
        fff02e9c5.jpg,3
        fffe98443.jpg,3
        ffff4eaa8.jpg,3
        ffffd67df.jpg,3
        00031f466.jpg,
        000418bfc.jpg,
        000789191.jpg,
            """)

        df = pd.read_csv(TESTDATA, sep=",")
        ds=image_datasets.MultiClassClassificationDataSet([],df,"ImageId","ClassId")
        ds.get_value=dummyImage

        self.assertEqual(len(ds.classes),4)

        encoded=ds.encode(ds, True)
        df['ClassId'] = df['ClassId']
        #df['ClassId'] = df['ClassId'].str.strip()
        rs=encoded['ClassId'].values==df['ClassId'].values
        pass