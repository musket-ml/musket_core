import unittest
from musket_core import dataset_analis as ds
from musket_core import datasets
import numpy as np

class TestStringMethods(unittest.TestCase):

    def test_simple(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                if item == 2:
                    raise ValueError()
                return datasets.PredictionItem(item, "H", 0)

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),1)
        pass

    def test_binary(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", [item%2])

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.outputTrials[0].numFirst,3)
        self.assertEqual(r.get_inputs_count(),1)
        self.assertEqual(r.get_outputs_count(), 1)
        pass

    def test_multi_output_binary(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", ([item%2],[item%3]))

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.get_inputs_count(), 1)
        self.assertEqual(r.get_outputs_count(), 2)
        #self.assertEqual(r.outputTrials[0].numFirst,3)
        pass

    def test_multiclass_binary(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", ([item%2,item%3]))

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.get_inputs_count(), 1)
        self.assertEqual(r.get_outputs_count(), 1)
        self.assertEqual(r.outputTrials[0].counts[0][0],3)
        pass

    def test_multiclass_binary2(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", (np.array([item%2,item%3])))

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.get_inputs_count(), 1)
        self.assertEqual(r.get_outputs_count(), 1)
        self.assertEqual(r.outputTrials[0].counts[0][0],3)
        pass

    def test_multiclass_binary3(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", (np.array([float(item%2),float(item%3==0)])))

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.get_inputs_count(), 1)
        self.assertEqual(r.get_outputs_count(), 1)
        self.assertEqual(r.outputTrials[0].counts[0][0],3)
        pass

    def test_multiclass_binary4(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", (np.array([float(item%2),float(item%3)])))

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.get_inputs_count(), 1)
        self.assertEqual(r.get_outputs_count(), 1)
        for i in r.outputTrials:
            if isinstance(i, ds.MultiClassification):
                self.assertTrue(False)
        pass

    def test_multiclass(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", (np.array([(item%3)])))

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.get_inputs_count(), 1)
        self.assertEqual(r.get_outputs_count(), 1)
        self.assertEqual(len(r.outputTrials),1)
        for i in r.outputTrials:
            self.assertEqual(i.isMulti,False)
        pass

    def test_multiclass2(self):
        class C(datasets.DataSet):
            def __init__(self):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, item):
                return datasets.PredictionItem(item, "H", (np.array([(item%3),6,7])))

        r = ds.DatasetTrial(C())
        self.assertEqual(len(r.errors),0)
        self.assertEqual(r.get_inputs_count(), 1)
        self.assertEqual(r.get_outputs_count(), 1)
        self.assertEqual(len(r.outputTrials), 2)
        for i in r.outputTrials:
            if isinstance(i,ds.MultiSetClassification):
                self.assertEqual(i.isMulti,True)
        pass