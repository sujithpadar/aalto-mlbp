import pandas as pd
import numpy as np


class LogisticRegression:

    def __init__(self,alpha=1e-4, niter=1e+4, tolerence=1e-5):

        '''
        :param alpha: learning rate
        :param niter: number of iterations before convergence
        :param tolerence: the delta loglikelihood for convergence
        '''

        self.alpha = alpha
        self.niter = niter
        self.tolerence = tolerence

    def train(self,data,target,features):
        '''
        :param data: input pandas dataframe
        :param target: the target label
        :param features: python list of explnatory features
        :return: weights for each feature
        '''

        self.data = data
        self.features = features
        self.target = target

        self.x = self.data[features]
        self.x['constant'] = 1      # add a constant vector
        self.features.insert(0, "constant")
        self.x = self.x[features]

        self.y = np.concatenate(self.data[target].values)

        self.weights = self.GradientDescent()
        return self.weights


    def sigmoid(self):
        '''
        :return: computes sigmoid for given data vector and weights
        '''

        g = 1/(1+np.exp(-self.x.dot(self.weights)))
        return g

    def logLikelihood(self):
        '''
        :return: compute log likelihood cost function
        '''

        prob = self.sigmoid()
        loglikelihood = -1*(np.sum(self.y*np.log(prob+1e-50)+(1-self.y)*(1-np.log(1-prob+1e-50))))
        return loglikelihood

    def GradientDescent(self):
        '''
        :return: computes weights gradient descent algorithm
        '''

        # initialize weights to zero
        self.weights = [0 for f in self.features]
        prob = self.sigmoid()
        logllprevious = self.logLikelihood()

        # compute gradient
        gradient = self.x.multiply(self.y - prob, axis=0).sum(axis=0)

        # update weights till maximum iterations
        for iteration in np.arange(0,self.niter):
            self.weights = self.weights + (self.alpha * gradient)
            logll = self.logLikelihood()

            # check for convergence
            if np.abs(logll - logllprevious) < self.tolerence:
                break

            # weight update
            logllprevious = logll
            prob = self.sigmoid()
            gradient = self.x.multiply(self.y - prob, axis=0).sum(axis=0)

        return self.weights

    def predict(self,newdata,type="raw",classthreshold=0.5):
        '''
        :param newdata: new data for which the prediction is made
        :param type: return prediction type : raw & class
        :return:
        '''

        newdata['constant'] = 1     # add a constant vector
        newdata = newdata[self.features]
        prediction = 1 / (1 + np.exp(-newdata.dot(self.weights)))

        # check return type
        if type == "class":
            prediction = (prediction >= classthreshold) * 1

        return prediction
asas
