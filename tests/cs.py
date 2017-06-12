# running Cuckoo Search to find best fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# svr and bts
from svr import SVRClassifier
from bts import bts_base, pathloss
from optimization.cuckoosearch import CuckooSearch

# base
X = bts_base[1.0]
y = pathloss[1]['plreal']

# SVRClassifir
svr_classifier = SVRClassifier(X, y)

def fitness(values):
    [C, gamma] = values
    
    # set new parameters
    svr_classifier.set_parameters(cost=C, e=0.1, y=gamma)
    
    # predict
    svr_y = svr_classifier.predict(10)

    # get error
    error = mean_squared_error(y, svr_y)

    return error


cs = CuckooSearch(2, fitness, [(0, 16), (0, 1)], num_nest=15, p=0.25)

[C, gamma] = cs.search(2)

print("Best values are C: {}, and gamma: {}". format(C, gamma))





