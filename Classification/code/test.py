import numpy as np
import pandas as pd
from itertools import chain,combinations,compress
from scipy.stats import itemfreq

import pandas as pd
import random


'''
Import data and declare features
'''
rawdata = pd.read_csv("/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/classification_dataset_training.csv")
target = ['rating']
features = ['but','good','place','food','great','very','service','back','really','nice',
            'love','little','ordered','first','much','came','went','try','staff','people',
            'restaurant','order','never','friendly','pretty','come','chicken','again','vegas',
            'definitely','menu','better','delicious','experience','amazing','wait','fresh','bad',
            'price','recommend','worth','enough','customer','quality','taste','atmosphere','however',
            'probably','far','disappointed']


# Test and training split
random.seed(1234)
trainvec = random.sample(range(0,rawdata.shape[0]),round(rawdata.shape[0]*0.7))
traindf = rawdata.loc[trainvec,]
validationdf = rawdata.loc[set(range(0,rawdata.shape[0]))-set(trainvec),]

x = traindf[features]
y = np.concatenate(traindf[target].values)


def DecisionTreeTrain(x,y,maxdepth = 3,depthl=0,depthr=0):
    # load the data and stopping criteria
    # recursively build tree till the max depth is reached or  the end nodes are reached on all the sub trees
    # list of dictionaries structure for trees
    print(depthl,depthr)
    tfeature = y.copy()
    tdata = x.copy()

    stree = buildClassificationSubTree(tdata,tfeature)

    # check the stop criteria
    if(stree['leftnode']['type'] == 'endnode' or depthl >= maxdepth):
        stree['leftnode']['ctree'] = None

    else:
        subdata = tdata.iloc[stree['leftnode']['index']]
        subtarget = tfeature[np.array(stree['leftnode']['index'])]
        stree['leftnode']['ctree'] =  DecisionTreeTrain(subdata,subtarget,maxdepth,depthl+1,depthr+1)

    if (stree['rightnode']['type'] == 'endnode' or depthr >= maxdepth):
        stree['rightnode']['ctree'] = None

    else:
        subdata = tdata.iloc[stree['rightnode']['index']]
        subtarget = tfeature[np.array(stree['rightnode']['index'])]
        stree['rightnode']['ctree'] = DecisionTreeTrain(subdata,subtarget,maxdepth,depthl+1,depthr+1)

    stree['leftnode']['index'] = None
    stree['rightnode']['index'] = None

    return stree


def buildClassificationSubTree(subx,suby,minnodesize=30):
    '''
    :param subx: data for subtree
    :param suby: target for subtree
    :return: returns a single node decision stump
    '''
    tfeature = suby.copy()
    tdata = subx.copy()

    # compute prior entropy
    priorentropy = entropy(tfeature)

    # initialize feature best split & entropy
    fname = tdata.columns
    fbestsplit = []
    fbestigr = []
    ftype = []

    # for every feature in the training set:
    for f in fname :
        # check the data type of the feature
        ffeature = tdata[f].values
        if np.unique(ffeature).size <= 10:
            # find split for discrete attributes
            # returns split criteria and split information gain
            ftype.append('discrete')
            figr = igrDiscrete(ffeature,tfeature,priorentropy)
        else:
            # find split for continuous attributes
            # returns split criteria and split information gain
            figr = igrNumeric(ffeature,tfeature,priorentropy)
            ftype.append('continuous')

        fbestigr.append(figr['maxigr'])
        fbestsplit.append(figr['comb'])

    # choose the best split
    pnodeigr = max(fbestigr)
    pnodelabel = fname[np.where(fbestigr == pnodeigr)[0][0]]
    pnodefeature = tdata[pnodelabel].values
    pnodedtype = ftype[np.where(fbestigr == max(fbestigr))[0][0]]

    # format the split criteria
    if(pnodedtype == 'discrete'):
        leftnodelabel = list(fbestsplit[np.where(fbestigr == max(fbestigr))[0][0]])
        leftfeatureindex = list(pval in leftnodelabel for pval in pnodefeature)
        leftnodecrosstab = itemfreq(list(compress(tfeature,leftfeatureindex)))

        rightnodelabel = list(set(np.unique(pnodefeature)) - set(leftnodelabel))
        rightfeatureindex = list(pval in rightnodelabel for pval in pnodefeature)
        rightnodecrosstab = itemfreq(list(compress(tfeature,rightfeatureindex)))

    else:
        leftnodelabel = list(fbestsplit[np.where(fbestigr == max(fbestigr))[0][0]])
        leftfeatureindex = list(pval <= leftnodelabel[0] for pval in pnodefeature)
        leftnodecrosstab = itemfreq(list(compress(tfeature,leftfeatureindex)))


        rightnodelabel = leftnodelabel
        rightfeatureindex = list(pval > leftnodelabel[0] for pval in pnodefeature)
        rightnodecrosstab = itemfreq(list(compress(tfeature,rightfeatureindex)))


    if(sum([nclass[1] for nclass in leftnodecrosstab]) <= minnodesize or leftnodecrosstab.shape[1] == 1):
        leftnodetype = "endnode"
    else:
        leftnodetype = "intermediatenode"
    if (sum([nclass[1] for nclass in rightnodecrosstab]) <= minnodesize or rightnodecrosstab.shape[1] == 1):
        rightnodetype = "endnode"
    else:
        rightnodetype = "intermediatenode"


    subtree = {'parent':{'label':pnodelabel,
                         'dtype':pnodedtype},
               'leftnode':{'splitc':leftnodelabel,
                           'index':leftfeatureindex,
                           'crosstab':leftnodecrosstab,
                           'type':leftnodetype},
               'rightnode':{'splitc':rightnodelabel,
                            'index':rightfeatureindex,
                            'crosstab':rightnodecrosstab,
                            'type':rightnodetype}}
    return subtree


def igrDiscrete(dfeature,tfeature,priorentropy):
    '''
    :param dfeature: discrete explanatory feature
    :param tfeature: target feature
    :param priorentropy: prior entropy
    :return: dict{'maxigr':max igr that can be achieved by this feature,
                  'maxigrcomb':tuple with elements in the best combination}
    '''
    # dfeature = x[features[1]].values
    # priorentropy = priorent
    # for each combinaiton of class, compute the information grain ratio
    comb = list(chain.from_iterable(combinations(np.unique(dfeature),n) for n in range(len(np.unique(dfeature))+1)))[1:-1]
    if len(comb) == 0:
        maxigrdict = {'maxigr': float('-inf'),
                      'comb': None}
    else:

        combigr = []
        for ccomb in comb:
            # idenitfy the subset and find the entropy
            classdivf = np.array([avalue in ccomb for avalue in dfeature])* 1
            # compute igr
            combigr.append(infogainratio(classdivf,tfeature,priorentropy))

        # return combination with highest infogainratio
        combigr = np.array(combigr)

        maxigrdict = {'maxigr':max(combigr),
                  'comb':comb[np.where(combigr == max(combigr))[0][0]]}

    return maxigrdict



def igrNumeric(nfeature,tfeature,priorentropy):
    '''
    :param nfeature: numeric explanatory feature
    :param tfeature: target feature
    :param priorentropy: prior entropy
    :return: dict{'maxigr':max igr that can be achieved by this feature,
                  'maxigrcomb':tuple with elements in the best combination}
    '''
    # nfeature = x[features[1]].values

    # identify cut points in the numeric feature
    comb = np.unique(nfeature)[:-1]

    combigr = []
    for ccomb in comb:
        # idenitfy the subset and find the entropy
        classdivf = np.array([avalue <= ccomb for avalue in nfeature])* 1
        # compute igr
        combigr.append(infogainratio(classdivf,tfeature,priorentropy))
    # return combination with highest infogainratio
    combigr = np.array(combigr)

    return {'maxigr':max(combigr),
            'comb':comb[np.where(combigr == max(combigr))[0][0]]}


def infogainratio(cfeat,tfeat,priorentropy):
    '''
    :param cfeat: explanatory features with each class represented as 0/1
    :param tfeat: target feature with each class represented as 0/1
    :param priorent: prio entropy - list with 'entropy' element as prior entropy
    :return: igr for the class feature
    '''

    # class proportions
    classunique, classprop = np.unique(cfeat, return_counts=True)
    classprop = [c / sum(classprop) for c in classprop]

    # idenitfy cross entropy using the above feature
    posteriorentropy  = np.sum([(entropy(tfeat[classunique[i] == cfeat])['entropy']) * classprop[i] for i in range(len(classunique))])

    infogain = priorentropy['entropy'] - posteriorentropy
    intrinsicvalue = entropy(cfeat)['entropy']

    return infogain/intrinsicvalue


def entropy(efeature):
    '''
    :param efeature: class/target feature
    :return: entropy for the feature
    '''
    uniqiue,counts = np.unique(efeature,return_counts=True)
    props = [classcount/np.sum(counts) for classcount in counts]
    entropy = np.sum([-(p * np.log2(p)) for p in props])
    return {'proirclasses' : uniqiue,
            'entropy'      : entropy,
            'props'        : props,
            'counts'       : counts}



dtree = DecisionTreeTrain(x,y)

def predict(newdata,type="raw"):
    '''
    :param newdata: data to predict on
    :param type: raw probabilities or class
    :return: predictions
    '''
    ndata = newdata[features].copy


