from .ClassificationSummary import ConfusionMatrix
import numpy as np
import random
from math import floor

def KfoldCV(model,data,target,features, k = 5, rseed = 1234):
    '''
    :param model: machine learning model class
    :param data: modeing data
    :param target: target vector name
    :param features: list of feature vecrot names
    :param k: number of folds
    :param rseed: random seed for the splits
    :return: ConfusionMatrix summary for the corss validation
    '''
    # get index for radom splits of k folds
    random.seed(rseed)
    rseq = np.array(random.sample(range(0, data.shape[0]),data.shape[0]))
    blist = np.arange(0,(data.shape[0]+1),floor(data.shape[0]/k))
    blist[-1] = data.shape[0]

    kidx = np.array([rseq[range(blist[i],blist[i+1])] for i in range(k)])

    # initialize vector for cross validation summary
    cvsummary = {'Accuracy' : 0,
               'Positive Pred Accuracy': 0,
               'Positive Pred Rate': 0,
               'Negative Pred Accuracy': 0,
               'Negative Pred Rate': 0}

    # run a loop across the k folds
    for i in np.arange(0,k):
        # keep 1 fold for validation and rest as training
        valdf = data.loc[kidx[i],]
        traindf = data.loc[set(range(0, data.shape[0])) - set(kidx[i]),]

        # train the model and get predicitons
        cmodel = model
        cweights = cmodel.train(traindf,target,features)
        traindf['pred'] = cmodel.predict(newdata=traindf, type="class")
        modelsumm = ConfusionMatrix(np.concatenate(traindf[target].values),traindf['pred'])

        # find average summary
        for metric in cvsummary.keys():
            cvsummary[metric] = cvsummary[metric] + (modelsumm[metric]/k)

    return cvsummary




