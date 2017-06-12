"""
Utiliza implementação SVR do sklearn para aproximar o Path Loss(Attenuation) dado base X e y
"""
# SVR, numpy
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

class SVRClassifier:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        # SVR
        self.clf = SVR(kernel='rbf', C=16)

        
    def set_parameters(self, cost=16, e=0.1, y=0.1):
        
        self.clf = SVR(kernel='rbf', C=cost, epsilon=e, gamma=1/2000)

            
    def predict(self, k_folds):
        
        predicts = cross_val_predict(self.clf, self.X, self.y, cv=k_folds)
    
        return predicts


    def score(self, k_folds):
        
        scores = cross_val_score(self.clf, self.X, self.y, cv=k_folds)
        
        return scores