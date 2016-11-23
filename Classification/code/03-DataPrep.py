import pandas as pd
import random

'''
Import data and declare features
'''
rawdf = pd.read_csv(
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

allfeatures = list(set(rawdf.columns.values) - set(['ID', 'rating']))

newfeatures = [feature for feature in allfeatures if feature.endswith("_m")]

'''
train and validation split
'''
# Test and training split
random.seed(1234)
trainvec = random.sample(range(0, rawdf.shape[0]), round(rawdf.shape[0] * 0.7))
traindf = rawdf.loc[trainvec,]
validationdf = rawdf.loc[set(range(0, rawdf.shape[0])) - set(trainvec),]
