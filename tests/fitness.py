"""
Fitness function
"""
# sklearn
from sklearn.metrics import mean_squared_error

# svr and bts
from svr import SVRClassifier
from bts import bts_base, pathloss

# base
X = bts_base[4.0]
y = pathloss[4]['plreal']

# SVRClassifir
svr_classifier = SVRClassifier(X, y)

def fitness(values):
    [C, ep] = values
    
    # set new parameters
    svr_classifier.set_parameters(cost=C, e=ep, y=1/2000)
    
    # predict
    svr_y = svr_classifier.predict(10)
    
    # get error
    error = mean_squared_error(y, svr_y)
    
    return error