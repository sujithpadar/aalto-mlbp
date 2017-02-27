import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,metrics
import pandas as pd
from pandas import DataFrame,Series
import random
from scipy.stats import norm
from sklearn import preprocessing
from patsy import dmatrices, ModelDesc, Term, LookupFactor,Poly

#from sklearn.cross_validation import cross_val_score

# Load the training data
rawdata = pd.read_csv(r'C:\Users\MY\Documents\Courses\MLBP\Term Project\aalto-mlbp\aalto-mlbp-master\Classification\data\regression_dataset_training.csv')
target = ['vote']
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

''' Make formula work
formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(c)]) for c in features])

y, x = dmatrices(formula, traindf, return_type="dataframe")
y = y.values.flatten()
'''

#### Creating a linear regression model on training data #########
x = traindf[features]
y = traindf[target]
#formula = ModelDesc([Term([LookupFactor('rating')])], [Term([LookupFactor(Poly(c))]) for c in features])
#y, x = dmatrices(formula, traindf, return_type="dataframe")
#y = y.values.flatten()



# Create linear regression object
linreg = linear_model.LinearRegression()

# Train the model using all the data

linreg.fit(x,y)
pred_y_lin = linreg.predict(x)
y['pred_y_lin'] = pred_y_lin


########### For bayesian classification ###############
Class = DataFrame(columns=['Mean','Variance'],index=np.unique(y[target]))
for i in np.unique(traindf[target]):
    Dummy = y[y['vote'] == i]
    Class['Mean'].loc[i] = np.mean(Dummy.pred_y_lin)
    Class['Variance'].loc[i] = np.var(Dummy.pred_y_lin)

################ For logistic Regression #################
expect = (y['vote'] >= y['pred_y_lin'])
#x['pred_y_lin'] = pred_y_lin - np.floor(pred_y_lin)
x['pred_y_lin'] = pred_y_lin - np.floor(pred_y_lin)
#x['pred_y_lin'] = pred_y_lin


#logreg = linear_model.LogisticRegressionCV(penalty='l1', tol=0.01, solver='liblinear')
logreg = linear_model.LogisticRegression()
logreg.fit(x[['pred_y_lin']], np.ravel(expect))


def bayesclass_predict(Class,model,data):
    x = data
    k = model.predict(x)

    df = DataFrame(index = Class.index.values,columns=x.index.values)
    for i in Class.index.values:
        df.loc[i] = norm.logpdf(x=np.ravel(k), loc=Class.Mean.ix[i], scale=Class.Variance.ix[i])

    condition = np.ravel([df.max() > -50])
    j =  np.round(np.ravel(k))
    j = j * (~condition)
    j = j + np.ravel(df.idxmax()) * condition

    return j

def logreg_predict(model_reg,model_class,data):
    x = data
    k = model_reg.predict(x)
    x['pred_y_lin'] = k - np.floor(k)
    #x['pred_y_lin'] = k
    y = model_class.predict(x[['pred_y_lin']])

    k = np.ravel(k)
    k = np.ceil(k * y) + np.floor(k * ~y)

    return k

def round_predict(model,data):
    x = data
    k = model.predict(x)
    k = np.round(k)

    return k

testdata = pd.read_csv(r'C:\Users\MY\Documents\Courses\MLBP\Term Project\aalto-mlbp\aalto-mlbp-master\Classification\data\regression_dataset_testing.csv')
testresult = pd.read_csv(r'C:\Users\MY\Documents\Courses\MLBP\Term Project\aalto-mlbp\aalto-mlbp-master\Classification\data\regression_dataset_testing_solution.csv')

op = logreg_predict(linreg,logreg,traindf[features])
print('Mean squared error for training data with logistic',metrics.mean_squared_error(traindf['vote'],op))

op = logreg_predict(linreg,logreg,validationdf[features])
print('Mean squared error for validation data with logistic',metrics.mean_squared_error(validationdf['vote'],op))

op = logreg_predict(linreg,logreg,testdata[features])
print('Mean squared error for test data with logistic',metrics.mean_squared_error(testresult['vote'],op))

op = bayesclass_predict(Class,linreg,traindf[features])
print('Mean squared error for training data with bayes',metrics.mean_squared_error(traindf['vote'],op))

op = bayesclass_predict(Class,linreg,validationdf[features])
print('Mean squared error for validation data with bayes',metrics.mean_squared_error(validationdf['vote'],op))

op = bayesclass_predict(Class,linreg,testdata[features])
print('Mean squared error for test data with bayes',metrics.mean_squared_error(testresult['vote'],op))

op = round_predict(linreg,traindf[features])
print('Mean squared error for training data with round',metrics.mean_squared_error(traindf['vote'],op))

op = round_predict(linreg,validationdf[features])
print('Mean squared error for validation data with round',metrics.mean_squared_error(validationdf['vote'],op))

op = round_predict(linreg,testdata[features])
print('Mean squared error for test data with round',metrics.mean_squared_error(testresult['vote'],op))



#plt.scatter(y['vote'],y['pred_final'])
#plt.show()







