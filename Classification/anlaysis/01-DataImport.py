import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .02-LogisticRegression.py import LogisticRegression


data = pd.read_csv("/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/data/classification_dataset_training.csv")
scaler = MinMaxScaler()

target = ["rating"]
features = ['but','good','place','food','great','very','service','back','really','nice','love','little','ordered','first','much','came','went','try','staff','people','restaurant','order','never','friendly','pretty','come','chicken','again','vegas','definitely','menu','better','delicious','experience','amazing','wait','fresh','bad','price','recommend','worth','enough','customer','quality','taste','atmosphere','however','probably','far','disappointed']

# data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

logreg = LogisticRegression()
logreg.train(data,target,features)
logreg.predict(newdata=data,type="class")