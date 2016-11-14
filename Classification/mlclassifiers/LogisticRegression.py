import numpy as np


class LogisticRegression:

    def __init__(self,alpha=1e-4, niter=1e+4, tolerence=1e-5,L2regular = False,lambreg = 0.01):

        '''
        :param alpha: learning rate
        :param niter: number of iterations before convergence
        :param tolerence: the delta loglikelihood for convergence
        :param L2regular: Boolean - for L2 regulairization cost
        :param lambreg: lambda for L2 regularization
        '''

        self.alpha = alpha
        self.niter = niter
        self.tolerence = tolerence
        self.regularization = L2regular
        self.lamb = lambreg

    def train(self,data,target,features = None):
        '''
        :param data: input pandas dataframe
        :param target: the target label
        :param features: python list of explnatory features
        :return: weights for each feature
        '''

        self.data = data.copy()
        self.features = features.copy()
        self.target = target.copy()

        self.x = self.data[self.features]
        self.x['constant'] = 1    # add a constant vector
        self.features.insert(0, "constant")
        self.x = self.x[self.features]

        self.y = np.concatenate(self.data[target].values)

        if self.regularization :
            self.weights = self.GradientDescent()
        else :
            self.weights = self.GradientDescentL2Regular()
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
        self.weights = np.array([0 for f in self.features])
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


    def logLikelihoodL2(self):
        '''
        :return: compute log likelihood cost function with L2 penalization
        '''

        prob = self.sigmoid()
        loglikelihoodL2 = (-1*(np.sum(self.y*np.log(prob+1e-50)+(1-self.y)*(1-np.log(1-prob+1e-50))))) + ((self.lamb/(2*self.x.shape[0]))*(self.weights.dot(self.weights)))
        return loglikelihoodL2

    def GradientDescentL2Regular(self):
        '''
        :return: computes weights gradient descent algorithm with L2 regularization
        '''

        # initialize weights to zero
        self.weights = np.array([0 for f in self.features])
        prob = self.sigmoid()
        logllprevious = self.logLikelihoodL2()

        # compute gradient
        gradient = self.x.multiply(self.y - prob, axis=0).sum(axis=0)

        # update weights till maximum iterations
        for iteration in np.arange(0,self.niter):
            self.weights = (self.weights*(1-(self.alpha*self.lamb/self.x.shape[0]))) + (self.alpha * gradient)
            logll = self.logLikelihoodL2()

            # check for convergence
            if np.abs(logll - logllprevious) < self.tolerence:
                break

            # weight update
            logllprevious = logll
            prob = self.sigmoid()
            gradient = self.x.multiply(self.y - prob, axis=0).sum(axis=0)

        return self.weights

    def predict(self, newdata, type="raw", classthreshold=0.5):
        '''
        :param newdata: new data for which the prediction is made
        :param type: return prediction type : raw & class
        :return:
        '''

        ndata = newdata.copy()

        ndata['constant'] = 1  # add a constant vector
        ndata = ndata[self.features]
        prediction = 1 / (1 + np.exp(-ndata.dot(self.weights)))

        # check return type
        if type == "class":
            prediction = (prediction >= classthreshold) * 1

        return prediction
