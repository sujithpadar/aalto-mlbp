import numpy as np
from itertools import chain, combinations, compress
from scipy.stats import itemfreq


class DecisionTree:
    def __init__(self, maxdepth=5, minnodesize=10, minnodeprop=0.01):

        self.maxdepth = maxdepth
        self.minnodesize = minnodesize
        self.minnodeprop = minnodeprop

    def train(self, data, target, features=None):
        '''
        :param data: input pandas dataframe
        :param target: the target label
        :param features: python list of explnatory features
        :return: tree for the features
        '''

        self.data = data.copy()
        self.features = features.copy()
        self.target = target.copy()

        self.x = self.data[self.features]
        self.y = np.concatenate(self.data[target].values)

        self.dtree = self.DecisionTreeTrain(self.x, self.y)

        return self.dtree

    def DecisionTreeTrain(self, xx, yy, depthl=0, depthr=0):
        # load the data and stopping criteria
        # recursively build tree till the max depth is reached or  the end nodes are reached on all the sub trees
        # list of dictionaries structure for trees

        tfeature = yy.copy()
        tdata = xx.copy()

        stree = self.buildClassificationSubTree(tdata, tfeature)

        if stree['parent']['label'] == None:
            stree = None

        else:
            # check the stop criteria
            if (stree['leftnode']['type'] == 'endnode' or depthl >= self.maxdepth):
                stree['leftnode']['ctree'] = None

            else:

                subdata = tdata.iloc[stree['leftnode']['index']]
                subtarget = tfeature[np.array(stree['leftnode']['index'])]
                stree['leftnode']['ctree'] = self.DecisionTreeTrain(subdata, subtarget, depthl + 1, depthr + 1)

            if (stree['rightnode']['type'] == 'endnode' or depthr >= self.maxdepth):
                stree['rightnode']['ctree'] = None

            else:

                subdata = tdata.iloc[stree['rightnode']['index']]
                subtarget = tfeature[np.array(stree['rightnode']['index'])]
                stree['rightnode']['ctree'] = self.DecisionTreeTrain(subdata, subtarget, depthl + 1, depthr + 1)

            stree['leftnode']['index'] = None
            stree['rightnode']['index'] = None

        return stree

    def buildClassificationSubTree(self, subx, suby):

        '''
        :param subx: data for subtree
        :param suby: target for subtree
        :return: returns a single node decision stump
        '''

        tfeature = suby.copy()
        tdata = subx.copy()

        # compute prior entropy
        priorentropy = self.entropy(tfeature)

        # initialize feature best split & entropy
        fname = tdata.columns
        fbestsplit = []
        fbestigr = []
        ftype = []

        # for every feature in the training set:
        for f in fname:
            # check the data type of the feature
            ffeature = tdata[f].values
            if np.unique(ffeature).size <= 1:
                # find split for discrete attributes
                # returns split criteria and split information gain
                ftype.append('discrete')
                figr = self.igrDiscrete(ffeature, tfeature, priorentropy)
            else:
                # find split for continuous attributes
                # returns split criteria and split information gain
                figr = self.igrNumeric(ffeature, tfeature, priorentropy)
                ftype.append('continuous')

            fbestigr.append(figr['maxigr'])
            fbestsplit.append(figr['comb'])


        # choose the best split
        pnodeigr = max(fbestigr)
        if max(fbestigr) == float('-inf'):
            subtree = {'parent': {'label': None,
                                  'dtype': None},
                       'leftnode': {'splitc': None,
                                    'index': None,
                                    'crosstab': None,
                                    'type': 'endnode'},
                       'rightnode': {'splitc': None,
                                     'index': None,
                                     'crosstab': None,
                                     'type': 'endnode'}}
        else:
            pnodelabel = fname[np.argmax(fbestigr)]
            pnodefeature = tdata[pnodelabel].values
            pnodedtype = ftype[np.argmax(fbestigr)]

            # format the split criteria
            if (pnodedtype == 'discrete'):

                leftnodelabel = list(fbestsplit[np.argmax(fbestigr)])
                leftfeatureindex = list(pval in leftnodelabel for pval in pnodefeature)
                leftnodecrosstab = itemfreq(list(compress(tfeature, leftfeatureindex)))

                rightnodelabel = list(set(np.unique(pnodefeature)) - set(leftnodelabel))
                rightfeatureindex = list(pval in rightnodelabel for pval in pnodefeature)
                rightnodecrosstab = itemfreq(list(compress(tfeature, rightfeatureindex)))

            else:
                leftnodelabel = [(fbestsplit[np.argmax(fbestigr)])]
                leftfeatureindex = list(pval <= leftnodelabel[0] for pval in pnodefeature)
                leftnodecrosstab = itemfreq(list(compress(tfeature, leftfeatureindex)))

                rightnodelabel = leftnodelabel
                rightfeatureindex = list(pval > leftnodelabel[0] for pval in pnodefeature)
                rightnodecrosstab = itemfreq(list(compress(tfeature, rightfeatureindex)))

            if sum([nclass[1] for nclass in leftnodecrosstab]) <= self.minnodesize or leftnodecrosstab.shape[1] == 1:
                leftnodetype = "endnode"
            else:
                leftnodetype = "intermediatenode"
            if sum([nclass[1] for nclass in rightnodecrosstab]) <= self.minnodesize or rightnodecrosstab.shape[1] == 1:
                rightnodetype = "endnode"
            else:
                rightnodetype = "intermediatenode"

            subtree = {'parent': {'label': pnodelabel,
                                  'dtype': pnodedtype},
                       'leftnode': {'splitc': leftnodelabel,
                                    'index': leftfeatureindex,
                                    'crosstab': leftnodecrosstab,
                                    'type': leftnodetype},
                       'rightnode': {'splitc': rightnodelabel,
                                     'index': rightfeatureindex,
                                     'crosstab': rightnodecrosstab,
                                     'type': rightnodetype}}

        return subtree

    def igrDiscrete(self, dfeature, tfeature, priorentropy):
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
        comb = list(
            chain.from_iterable(combinations(np.unique(dfeature), n) for n in range(len(np.unique(dfeature)) + 1)))[
               1:-1]
        if len(comb) == 0:
            maxigrdict = {'maxigr': float('-inf'),
                          'comb': None}
        else:

            combigr = []
            for ccomb in comb:
                # idenitfy the subset and find the entropy
                classdivf = np.array([avalue in ccomb for avalue in dfeature]) * 1

                if (np.sum(classdivf) / classdivf.shape[0]) <= self.minnodeprop or (
                            np.sum(classdivf) / classdivf.shape[0]) >= (
                            1 - self.minnodeprop):
                    classdivf = classdivf * 0

                # compute igr
                combigr.append(self.infogainratio(classdivf, tfeature, priorentropy))

            # return combination with highest infogainratio
            combigr = np.array(combigr)

            maxigrdict = {'maxigr': max(combigr),
                          'comb': comb[np.where(combigr == max(combigr))[0][0]]}

        return maxigrdict

    def igrNumeric(self, nfeature, tfeature, priorentropy):
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
            classdivf = np.array([avalue <= ccomb for avalue in nfeature]) * 1
            if (np.sum(classdivf) / classdivf.shape[0]) <= self.minnodeprop or (
                        np.sum(classdivf) / classdivf.shape[0]) >= (
                        1 - self.minnodeprop):
                classdivf = classdivf * 0
            # compute igr
            combigr.append(self.infogainratio(classdivf, tfeature, priorentropy))
        # return combination with highest infogainratio
        combigr = np.array(combigr)

        return {'maxigr': max(combigr),
                'comb': comb[np.where(combigr == max(combigr))[0][0]]}

    def infogainratio(self, cfeat, tfeat, priorentropy):
        '''
        :param cfeat: explanatory features with each class represented as 0/1
        :param tfeat: target feature with each class represented as 0/1
        :param priorentropy: prio entropy - list with 'entropy' element as prior entropy
        :return: igr for the class feature
        '''

        # class proportions
        classunique, classprop = np.unique(cfeat, return_counts=True)
        classprop = [c / sum(classprop) for c in classprop]

        # idenitfy cross entropy using the above feature
        posteriorentropy = np.sum(
            [(self.entropy(tfeat[classunique[i] == cfeat])['entropy']) * classprop[i] for i in range(len(classunique))])

        infogain = priorentropy['entropy'] - posteriorentropy
        intrinsicvalue = self.entropy(cfeat)['entropy']

        return infogain / (intrinsicvalue + 1e-100)

    def entropy(self, efeature):
        '''
        :param efeature: class/target feature
        :return: entropy for the feature
        '''
        uniqiue, counts = np.unique(efeature, return_counts=True)
        props = [classcount / np.sum(counts) for classcount in counts]
        entropy = np.sum([-(p * np.log2(p)) for p in props])
        return {'proirclasses': uniqiue,
                'entropy': entropy,
                'props': props,
                'counts': counts}

    def predict(self, newdata, type="raw"):
        '''
        :param newdata: data to predict on
        :param type: raw probabilities or class
        :return: predictions
        '''

        ndata = newdata[self.features].copy()
        pred = np.zeros(ndata.shape[0])
        for i in np.arange(0, ndata.shape[0]):
            nd = ndata.iloc[i, :]
            encrosstab = self.getPredictionEndBucket(self.dtree, nd)
            if (encrosstab.shape[0] == 1):
                if encrosstab[0][0] == 0:
                    pred[i] = 0
                else:
                    pred[i] = 1
            else:
                pred[i] = encrosstab[1][1] / np.sum(encrosstab[0][1] + encrosstab[1][1])

        if (type == "class"):
            poprate = sum(self.y) / self.y.shape[0]
            pred = (pred >= poprate) * 1

        return pred

    def getPredictionEndBucket(self, ttree, ndata):
        if ttree['parent']['dtype'] == 'discrete':
            if (ndata[ttree['parent']['label']] in ttree['rightnode']['splitc']):
                branch = 'rightnode'
            else:
                branch = 'leftnode'

        else:
            if (ndata[ttree['parent']['label']] > ttree['rightnode']['splitc']):
                branch = 'rightnode'
            else:
                branch = 'leftnode'

        if ttree[branch]['ctree'] == None:
            return ttree[branch]['crosstab']
        else:
            subtree = ttree[branch]['ctree']
            return self.getPredictionEndBucket(subtree, ndata)
