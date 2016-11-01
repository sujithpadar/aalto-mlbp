
from mlclassifiers import LogisticRegression,ConfusionMatrix,KfoldCV

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



'''
model 1 - logistic regression
'''
# train model
logreg = LogisticRegression()
logreg.train(traindf,target,features)

# prediction on train and test data
traindf['pred1'] = logreg.predict(newdata=traindf,type="class")
validationdf['pred1'] = logreg.predict(newdata=validationdf,type="class")

# classification summary
ConfusionMatrix(traindf['rating'],traindf['pred1'])
ConfusionMatrix(validationdf['rating'],validationdf['pred1'])

# crossvalidation summary
KfoldCV(model = logreg,data = rawdata,target = target,features = features,k = 5)

'''
model 2 - logitic regression with L2 regularization
'''
# train model
logreg = LogisticRegression(L2regular=True,lambreg=0.1)
logreg.train(traindf,target,features)

# prediction on train and test data
traindf['pred2'] = logreg.predict(newdata=traindf,type="class")
validationdf['pred2'] = logreg.predict(newdata=validationdf,type="class")

# classification summary
ConfusionMatrix(traindf['rating'],traindf['pred2'])
ConfusionMatrix(validationdf['rating'],validationdf['pred2'])

# crossvalidation summary
KfoldCV(model = logreg,data = rawdata,target = target,features = features, k = 5)

