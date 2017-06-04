"""
Utiliza implementação SVR do sklearn para aproximar o Path Loss(Attenuation) dado base X e y
"""
# SVR, numpy
from sklearn.svm import SVR
import numpy as np


def training_svr(X, y):
    """
    :param X training data base:
    :param y target value:
    :return SVR object fit for the database X and y:
    """
    
    #  first check the array size
    if len(X) != len(y):
        raise ValueError("Tamanho da base de dados X e y são diferentes.")

    # simple SVR
    clf = SVR(C=1.0, epsilon=0.1)
    clf.fit(X, y)
    
    return clf


def predict(clf, X):
    """
    :param clf - training svr:
    :param y - databse for prediction:
    :return: predict values
    """

    svr_y = clf.predict(X)
    
    return svr_y