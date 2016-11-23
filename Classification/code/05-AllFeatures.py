execfile("code/03-DataPrep.py")

from patsy import dmatrices, ModelDesc, Term, LookupFactor
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
import numpy as np

'''
model 1 - logistic regression with L1 regularization
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in allfeatures])

y, x = dmatrices(formula, rawdf, return_type="dataframe")
y = y.values.flatten()

logreg = linear_model.LogisticRegression(C=0.1, penalty='l1', tol=0.01)

logreg.fit(x, y)
scores = cross_val_score(logreg, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))

# feature selection using best model from cross validation and get the best features
fslogreg = linear_model.LogisticRegressionCV(penalty='l1', solver='liblinear')
fslogreg.fit(x, y)

fsmodel = SelectFromModel(fslogreg, prefit=True)
x_new = fsmodel.transform(x)
x_new.shape

coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(fslogreg.coef_).flatten()})
nflist = coeffdf[coeffdf.coeff != 0].feature.values.tolist()
print(len(nflist))

'''
L2 regularisation on new features
'''
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in nflist])
y, x = dmatrices(formula, rawdf, return_type="dataframe")
y = y.values.flatten()

logreg = linear_model.LogisticRegressionCV(penalty='l2')

# train model
logreg.fit(x, y)
coeffdf = pd.DataFrame({'feature': x.columns, 'coeff': np.transpose(logreg.coef_).flatten()})
print(logreg.C_)

scores = cross_val_score(logreg, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
