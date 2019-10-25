import unittest
from musket_core import coders
import numpy as np
import pandas as pd
import os
import math

fl=__file__

fl=os.path.dirname(fl)
class TestCoders(unittest.TestCase):

    def test_binary_num(self):
        a=np.array([0,1,0,1])
        bc=coders.get_coder("binary",a, None)
        self.assertEqual(bc[0], 0, "should be zero")
        self.assertEqual(bc[1], 1, "should be one")
        v=bc._decode(np.array([0.6]))
        self.assertEqual(v, 1, "should be one")
        v=bc._decode(np.array([0.2]))
        self.assertEqual(v, 0, "should be zero")
        pass
    def test_binary_str(self):
        a=np.array(["0","1","0","1"])
        bc=coders.get_coder("binary",a, None)
        self.assertEqual(bc[0], 0, "should be zero")
        self.assertEqual(bc[1], 1, "should be one")
        v=bc._decode(np.array([0.6]))
        self.assertEqual(v, "1", "should be one")
        v=bc._decode(np.array([0.2]))
        self.assertEqual(v, "0", "should be zero")
        pass
    def test_binary_str2(self):
        a=np.array(["","1","","1"])
        bc=coders.get_coder("binary",a, None)
        self.assertEqual(bc[0], 0, "should be zero")
        self.assertEqual(bc[1], 1, "should be one")
        v=bc._decode(np.array([0.6]))
        self.assertEqual(v, "1", "should be one")
        v=bc._decode(np.array([0.2]))
        self.assertEqual(v, "", "should be zero")
        pass
    def test_binary_bool(self):
        a=np.array([True,False,True,False])
        bc=coders.get_coder("binary",a, None)
        self.assertEqual(bc[0], 1, "should be zero")
        self.assertEqual(bc[1], 0, "should be one")
        v=bc._decode(np.array([0.6]))
        self.assertEqual(v, True, "should be one")
        v=bc._decode(np.array([0.2]))
        self.assertEqual(v, False, "should be zero")
        pass
    
    def test_categorical_num(self):
        a=np.array([0,1,2,1])
        bc=coders.get_coder("categorical_one_hot",a, None)
        
        self.assertEqual(bc[0][0], True, "should be zero")
        self.assertEqual(bc[0][1], False, "should be one")
        v=bc._decode(np.array([0.3,0.4,0.45]))
        self.assertEqual(v, 2, "should be one")
        v=bc._decode(np.array([0.2,0.1,0.1]))
        self.assertEqual(v, 0, "should be zero")
        pass
    
    def test_categorical_str(self):
        a=np.array(["a","b","c","b"])
        bc=coders.get_coder("categorical_one_hot",a, None)
        
        self.assertEqual(bc[0][0], True, "should be zero")
        self.assertEqual(bc[0][1], False, "should be one")
        v=bc._decode(np.array([0.3,0.4,0.45]))
        self.assertEqual(v, "c", "should be one")
        v=bc._decode(np.array([0.2,0.1,0.1]))
        self.assertEqual(v, "a", "should be zero")
        pass
    
    def test_categorical_str2(self):
        a=np.array(["","b","c","b"])
        bc=coders.get_coder("categorical_one_hot",a, None)
        
        self.assertEqual(bc[0][0], True, "should be zero")
        self.assertEqual(bc[0][1], False, "should be one")
        v=bc._decode(np.array([0.3,0.4,0.45]))
        self.assertEqual(v, "c", "should be one")
        v=bc._decode(np.array([0.2,0.1,0.1]))
        self.assertEqual(v, "", "should be zero")
        pass
    
    def test_categorical_pd(self):
        a=np.array([math.nan,1,2,1])
        bc=coders.get_coder("categorical_one_hot",a, None)
        
        self.assertEqual(bc[0][2], True, "should be zero")
        self.assertEqual(bc[0][1], False, "should be one")
        v=bc._decode(np.array([0.3,0.4,0.45]))
        self.assertEqual(math.isnan(v),True, "should be one")
        v=bc._decode(np.array([0.2,0.1,0.1]))
        self.assertEqual(v, 1, "should be zero")
        pass
    
    def test_multiclass(self):
        a=np.array(["1 2","0 2","0",""])
        bc=coders.get_coder("multi_class",a, None)
        
        val=bc[0]
        self.assertEqual((val==np.array([False,True,True])).sum(), 3,"Fixing format")
        for i in range(len(a)):
            val=bc[i]
            r=bc._decode(val)
            self.assertEqual(r, a[i], "Decoding should work also")
        pass
    
    def test_multiclass1(self):
        a=np.array(["1_2","0_2","0",""])
        bc=coders.get_coder("multi_class",a, None)
        
        val=bc[0]
        self.assertEqual((val==np.array([False,True,True])).sum(), 3,"Fixing format")
        for i in range(len(a)):
            val=bc[i]
            r=bc._decode(val)
            self.assertEqual(r, a[i], "Decoding should work also")
        pass
    
    def test_multiclass2(self):
        a=np.array(["1","","",""])
        bc=coders.get_coder("multi_class",a, None)
        
        val=bc[0]
        self.assertEqual((val==np.array([True])).sum(), 1,"Fixing format")
        for i in range(len(a)):
            val=bc[i]
            r=bc._decode(val)
            self.assertEqual(r, a[i], "Decoding should work also")
        pass