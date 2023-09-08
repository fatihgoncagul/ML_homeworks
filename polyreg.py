"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np
from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=4)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """
        Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        n = len(X)
        result = np.empty((n, degree))

        for i in range(0, n):
            x = X[i]
            for j in range(0, degree):
                result[i, j] = x ** (j + 1)

        return result


        
    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        n = len(X)
        X = self.polyfeatures(X, self.degree)
        self.zScore(X)
        X = self.zScore(X)
        
        


        # Add 1 to beginning of X
        X = np.c_[np.ones([n, 1]), X];

        n,d = X.shape

        # construct reg matrix
        regMatrix = self.regLambda * np.eye(d)
        regMatrix[0,0] = 0

        self.theta = np.linalg.pinv(X.T.dot(X) + regMatrix).dot(X.T).dot(y);





        n, d = X.shape

        result = np.zeros([n, d])

        for i in range(0, n):

            for j in range(0, d):
                if self.transformation[j, 1] == 0:
                    result[i, j] = 0
                else:
                    result[i, j] = (X[i, j] - self.transformation[j, 0])/self.transformation[j, 1]

        return result
       
    def zScore(X,rtn_ms=False):
    
     mu     = np.mean(X,axis=0)  
     sigma  = np.std(X,axis=0)
     X_norm = (X - mu)/sigma      

     if rtn_ms:
        return(X_norm, mu, sigma)
     else:
        return(X_norm)   
        

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n = len(X)

        X = self.polyfeatures(X, self.degree)
        X = self.zScore(X)
        
        # add 1s column
        X = np.c_[np.ones([n, 1]), X]

        

        return X.dot(self.theta)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    return np.square(np.subtract(a,b)).mean()


@problem.tag("hw1-A", start_line=5)
def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
      
    
    for i in range(0, n):
        model = PolynomialRegression(degree=degree, regLambda=regLambda)  
        model.fit(Xtrain[0:(i + 1)], Ytrain[0:(i + 1)])

        errorTrain[i] = computeError(model.predict(Xtrain[0:(i+1)]), Ytrain[0:(i+1)])
        errorTest[i] = computeError(model.predict(Xtest), Ytest)
        
    

    return (errorTrain, errorTest)


def computeError(calculated, actual):
        n = len(calculated)

        total = 0
        for i in range(0, n):
            total += (calculated[i] - actual[i]) ** 2

        return total/n