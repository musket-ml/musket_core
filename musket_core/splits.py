import random

import numpy as np
from sklearn import model_selection as ms

from musket_core.datasets import dataset_classes, SubDataSet


class IndicesDistribution:
    def __init__(self, folds, holdout):
        self.folds = folds
        if holdout is not None:
            self.holdout = holdout
        else:
            self.holdout = []

def distribute(ids:list,classes:list,folds:int,seed,extractHoldout:bool):

    for ind1 in range(len(classes)):
        c1 = set(classes[ind1])
        for ind2 in range(c1+1,len(classes)):
            c2 = classes[ind2]
            if len(c1.intersection(c2)) > 0:
                raise ValueError(f'Classes {c1} and {c2} have nonempty intersection')


    actualFolds = folds
    if extractHoldout:
        actualFolds = actualFolds + 1

    positives = []
    negatives = []

    trainIDs = set(ids)
    for cl in classes:
        trainIDs = trainIDs.difference(cl)

    trainIDs = np.array(list(trainIDs),dtype = np.int)

    classIndices = []
    currentSeed = random.getstate()
    random.seed(seed)
    for cl in classes:
        indices = list(range(0, len(cl)))
        random.shuffle(indices)
        shuffled = np.array(cl, dtype=np.int)[indices]
        classIndices.append(shuffled)

    random.seed(currentSeed)

    classFolds = []
    for ci in classIndices:
        _folds = np.array_split(ci, actualFolds)
        classFolds.append(_folds)

    actualValidationFolds = []
    for ind in range(0, actualFolds):
        avf = np.array([],dtype=np.int)
        for cf in classFolds:
            avf = np.concatenate((avf, ci))
        actualValidationFolds.append(avf)

    for ind1 in range(0, actualFolds):
        f1 = actualValidationFolds[ind1]
        for ind2 in range(ind1+1, actualFolds):
            f2 = actualValidationFolds[ind2]
            if len(set(f1).intersection(f2)) != 0:
                raise ValueError(f'Validation folds {f1} and {f2} have nonempty intersection')

    distributionsCount = 1
    distributions = []
    if extractHoldout:
        distributionsCount = actualFolds

    for dInd in range(0,distributionsCount):

        foldsValidation = actualValidationFolds.copy()
        holdout = None
        if extractHoldout:
            holdout = foldsValidation.pop(dInd)

        foldsTrain = (np.zeros((folds, len(trainIDs)), dtype = np.int) + np.array(list(trainIDs), dtype = np.int)).tolist()
        for shift in range(1, folds):
            perm = np.roll(foldsValidation, shift, axis = 0)
            for ind in range(0, folds):
                foldsTrain[ind] = np.concatenate((foldsTrain[ind], perm[ind]))

        for ind1 in range(0, folds):
            f1 = foldsTrain[ind1]
            for ind2 in range(ind1+1, folds):
                f2 = foldsTrain[ind2]
                expectedLength = len(trainIDs)
                for ind3 in range(0, folds):
                    if ind3 == ind1 or ind3 == ind2:
                        continue
                    expectedLength = expectedLength + len(foldsValidation[ind3])
                actualLength = len(set(f1).intersection(f2))
                if actualLength != expectedLength:
                    raise ValueError(f'Train folds {f1} and {f2} have {actualLength} while it should be {expectedLength}')

        foldsList = []
        for ind in range(0, folds):
            foldsList.append((foldsTrain[ind],foldsValidation[ind]))

        distribution = IndicesDistribution(foldsList,holdout)
        distributions.append(distribution)

    if extractHoldout:
        for dInd in range(0,len(distributions)):
            distrib = distributions[dInd]
            dHoldout = set(distribution.holdout.tolist())
            dFolds = distribution.folds
            for fInd in range(0,len(dFolds)):
                train, validation = dFolds[fInd]
                if len(dHoldout.intersection(train.tolist())) != 0:
                    raise ValueError(f'Holdout {dInd} and train {fInd} have nonempty intersection')
                if len(dHoldout.intersection(validation.tolist())) != 0:
                    raise ValueError(f'Holdout {dInd} and validation {fInd} have nonempty intersection')

    return distributions


def split(ds,testSplit,testSplitSeed,stratified=False,groupFunc=None):

    rn=list(range(0,len(ds)))
    if stratified:
        data_classes = dataset_classes(ds, groupFunc)
        vals=ms.StratifiedShuffleSplit(4,testSplit,random_state=testSplitSeed).split(rn,data_classes)
        for v in vals:
            return SubDataSet(ds, v[0]), SubDataSet(ds,v[1])

    random.seed(testSplitSeed)
    random.shuffle(rn)
    dm=round(len(ds)-len(ds)*testSplit)
    return SubDataSet(ds,rn[:dm]),SubDataSet(ds,rn[dm:])