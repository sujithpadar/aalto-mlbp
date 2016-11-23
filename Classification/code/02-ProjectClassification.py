from mlclassifiers import LogisticRegression, ConfusionMatrix, KfoldCV, DecisionTree
import pandas as pd
import numpy as np
import random
from patsy import dmatrices, ModelDesc, Term, LookupFactor
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

'''
Import data and declare features
'''
rawdata = pd.read_csv(
    "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/train.csv")
testdf = pd.read_csv(
    "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/test.csv")
target = ['rating']
orgfeatures = ['but', 'good', 'place', 'food', 'great', 'very', 'service', 'back', 'really', 'nice',
               'love', 'little', 'ordered', 'first', 'much', 'came', 'went', 'try', 'staff', 'people',
               'restaurant', 'order', 'never', 'friendly', 'pretty', 'come', 'chicken', 'again', 'vegas',
               'definitely', 'menu', 'better', 'delicious', 'experience', 'amazing', 'wait', 'fresh', 'bad',
               'price', 'recommend', 'worth', 'enough', 'customer', 'quality', 'taste', 'atmosphere', 'however',
               'probably', 'far', 'disappointed']

allfeatures = list(set(rawdata.columns.values) - set(['ID', 'rating']))

'''
train and validation split
'''
# Test and training split
random.seed(1234)
trainvec = random.sample(range(0, rawdata.shape[0]), round(rawdata.shape[0] * 0.7))
traindf = rawdata.loc[trainvec,]
validationdf = rawdata.loc[set(range(0, rawdata.shape[0])) - set(trainvec),]

'''
model 1 - logistic regression with L1 regularization
'''
from sklearn import linear_model

formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in allfeatures])

y, x = dmatrices(formula, traindf, return_type="dataframe")
y = y.values.flatten()

logreg = linear_model.LogisticRegression(C=0.1, penalty='l1', tol=0.01)
logreg = linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear')

# train model
logreg.fit(x, y)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))

# prediction on train and test data
traindf['pred_LRL1'] = logreg.predict(traindf[features])
validationdf['pred_LRL1'] = logreg.predict(validationdf[features])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL1'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL1'])

scores = cross_val_score(logreg, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


'''
L2 regularisation on new features
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in nflist])
y, x = dmatrices(formula, traindf, return_type="dataframe")
y = y.values.flatten()

logreg = linear_model.LogisticRegressionCV(penalty='l2')

# train model
logreg.fit(x, y)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
print(logreg.C_)

# prediction on train and test data
traindf['pred_LRL2'] = logreg.predict(traindf[nflist])
validationdf['pred_LRL2'] = logreg.predict(validationdf[nflist])
testdf['pred_LRL2'] = logreg.predict(testdf[nflist])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL2'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL2'])
ConfusionMatrix(testdf['rating'], testdf['pred_LRL2'])

scores = cross_val_score(logreg, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
L2 regularisation on origirnal features
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in orgfeatures])
y, x = dmatrices(formula, traindf, return_type="dataframe")
y = y.values.flatten()

logreg = linear_model.LogisticRegressionCV(penalty='l2')

# train model
logreg.fit(x, y)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
print(logreg.C_)

# prediction on train and test data
traindf['pred_LRL2'] = logreg.predict(traindf[nflist])
validationdf['pred_LRL2'] = logreg.predict(validationdf[nflist])
testdf['pred_LRL2'] = logreg.predict(testdf[nflist])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL2'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL2'])
ConfusionMatrix(testdf['rating'], testdf['pred_LRL2'])

scores = cross_val_score(logreg, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


'''
L2 regularisation normal
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in features])
y, x = dmatrices(formula, traindf, return_type="dataframe")

logreg = linear_model.LogisticRegression(penalty='l2')

# train model
logreg.fit(x, y)
print(logreg.intercept_)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})

# prediction on train and test data
traindf['pred_LRL2'] = logreg.predict(traindf[features])
validationdf['pred_LRL2'] = logreg.predict(validationdf[features])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL2'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL2'])

'''
run L1 on interaction features and identify best features and add them to L2
'''
interactionfeatures = [col for col in traindf.columns if col.startswith('but_') or col.contains('however_')]
# interactionfeatures = [col for col in traindf.columns if 'but' in col or 'however' in col]
# interactionfeatures = list(set(interactionfeatures) - set(['but','however']))

formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in interactionfeatures])
y, x = dmatrices(formula, traindf, return_type="dataframe")

logreg = linear_model.LogisticRegression(C=0.1, penalty='l1', tol=0.01)

# train model
logreg.fit(x, y)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))

# prediction on train and test data
traindf['pred_LRL1'] = logreg.predict(traindf[interactionfeatures])
validationdf['pred_LRL1'] = logreg.predict(validationdf[interactionfeatures])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL1'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL1'])

'''
L2 with new added interaction features
'''
aflist = nflist + orgfeatures
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in aflist])
y, x = dmatrices(formula, traindf, return_type="dataframe")

logreg = linear_model.LogisticRegression(penalty='l1')

# train model
logreg.fit(x, y)
print(logreg.intercept_)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})

# prediction on train and test data
traindf['pred_LRL2'] = logreg.predict(traindf[aflist])
validationdf['pred_LRL2'] = logreg.predict(validationdf[aflist])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL2'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL2'])

'''
cross validation
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in features])
y, x = dmatrices(formula, traindf, return_type="dataframe")

logreg = linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear')

# train model
logreg.fit(x, y)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))


avg_score = np.mean(
    np.array([cvscore[np.where(np.array(logreg.Cs_) == np.array(logreg.C_))[0][0]] for cvscore in logreg.scores_[1.0]]))

# prediction on train and test data
traindf['pred_LRL1'] = logreg.predict(traindf[features])
validationdf['pred_LRL1'] = logreg.predict(validationdf[features])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL1'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL1'])

'''
cross validation
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in features])
y, x = dmatrices(formula, rawdata, return_type="dataframe")

logreg = linear_model.LogisticRegression(C=0.1, penalty='l1', tol=0.01)

# train model
logreg.fit(x, y)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))

# prediction on train and test data
traindf['pred_LRL1'] = logreg.predict(traindf[features])
validationdf['pred_LRL1'] = logreg.predict(validationdf[features])

# classification summary
ConfusionMatrix(traindf['rating'], traindf['pred_LRL1'])
ConfusionMatrix(validationdf['rating'], validationdf['pred_LRL1'])

scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
