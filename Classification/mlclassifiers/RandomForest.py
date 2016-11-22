import numpy as np
import pandas as pd
import random
from mlclassifiers import DecisionTree, ConfusionMatrix

# implement random forest
# take random number of features
# divide data into random 70:30 split
# predict on the 30 set and keep appending all the preictions
rawdata = pd.read_csv(
    "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/classification_dataset_training.csv")
target = ['rating']
features = ['but', 'good', 'place', 'food', 'great', 'very', 'service', 'back', 'really', 'nice',
            'love', 'little', 'ordered', 'first', 'much', 'came', 'went', 'try', 'staff', 'people',
            'restaurant', 'order', 'never', 'friendly', 'pretty', 'come', 'chicken', 'again', 'vegas',
            'definitely', 'menu', 'better', 'delicious', 'experience', 'amazing', 'wait', 'fresh', 'bad',
            'price', 'recommend', 'worth', 'enough', 'customer', 'quality', 'taste', 'atmosphere', 'however',
            'probably', 'far', 'disappointed']


def trainRandomForest(rawdata, target, features, ntrees=50, nfeatures=5, maxdepth=10, minnodesize=10, minnodeprop=0.01):
    rawdata['rowindex'] = np.arange(0, rawdata.shape[0])

    data = rawdata.copy()
    features = features.copy()
    target = target.copy()

    # build tree
    randomforesttrees = {}
    rftestpred = {}

    for itree in np.arange(0, 10):
        # test and training split
        dsplitvec = random.sample(range(0, data.shape[0]), round(data.shape[0] * 0.7))
        itrainddata = data.loc[dsplitvec,]
        itestdata = data.loc[set(range(0, data.shape[0])) - set(dsplitvec),]

        # select random feature set
        fsplitvec = random.sample(range(0, len(features)), nfeatures)
        ifeatures = [features[f] for f in fsplitvec]

        dtree = DecisionTree(maxdepth=maxdepth, minnodesize=minnodesize, minnodeprop=minnodeprop)
        randomforesttrees[itree] = dtree.train(itrainddata, target, ifeatures)
        rftestpred[itree] = pd.DataFrame({'idx': np.array(itestdata['rowindex']),
                                          'pred': dtree.predict(itestdata, "class")})

    # aggregate predictions on the test data
    # create a prediction feature with a list of count of zeors and ones
    # map these to respective idx
    preddf = pd.DataFrame({'rowindex': np.arange(0, data.shape[0]),
                           'count0': np.zeros(data.shape[0]),
                           'count1': np.zeros(data.shape[0])})

    for itree in np.arange(0, 10):
        pdf = rftestpred[itree]

        preddf = preddf.merge(pdf, how='left',
                              left_on='rowindex', right_on='idx', suffixes=('', '_'))

        preddf['count0'] = preddf['count0'] + pd.Series((preddf['pred'] - 1) * (-1)).replace(to_replace=float('nan'),
                                                                                             value=0)
        preddf['count1'] = preddf['count1'] + pd.Series(preddf['pred']).replace(to_replace=float('nan'), value=0)
        preddf.drop(['idx', 'pred'], axis=1, inplace=True)

    preddf['pred'] = (preddf['count0'] < preddf['count1']) * 1

    data = data.merge(preddf.filter(['rowindex', 'pred']), how='left',
                      left_on='rowindex', right_on='rowindex', suffixes=('', '_'))

    data.drop('rowindex', axis=1, inplace=True)

    return randomforesttrees, data


trftrees, ptdata = trainRandomForest(rawdata, target, features, ntrees=1000, nfeatures=10, maxdepth=10, minnodesize=10,
                                     minnodeprop=0.01)
print(ConfusionMatrix(ptdata['rating'], ptdata['pred']))
