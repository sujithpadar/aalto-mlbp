import numpy as np


class LogisticRegression:

    def __init__(self,splitcriteria = "entropy",maxdepth=None,minnodesize=10):

        self.splitcriteria = splitcriteria
        self.maxdepth = maxdepth
        self.minnodesize = minnodesize


    def train(self,data,target,features = None):
        '''
        :param data: input pandas dataframe
        :param target: the target label
        :param features: python list of explnatory features
        :return: tree for the features
        '''

        self.data = data.copy()
        self.features = features.copy()
        self.target = target.copy()





