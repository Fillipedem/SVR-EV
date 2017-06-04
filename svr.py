"""
Utiliza implementação SVR do sklearn para aproximar o Path Loss(Attenuation)
"""
# SVR, numpy
from sklearn.svm import SVR
import numpy as np
# database
from database import erbs_header, data_erbs, medicoes_header, data_medicoes


# data medicoes
pos = medicoes_header.index('PLreal')

# training data X
X_header = medicoes_header[:pos] + medicoes_header[pos + 1:]
X = np.delete(data_medicoes, pos, 1)

# target data y
Y_header = ['PLreal']
y = data_medicoes[:, pos]

# simple SVR
clf = SVR(C=1.0, epsilon=0.1)
clf.fit(X, y)