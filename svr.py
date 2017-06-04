"""
Utiliza implementação SVR do sklearn para aproximar o Path Loss(Attenuation)
"""
# SVR, numpy
import sklearn.svm
import numpy as np
# database
from database import erbs_header, data_erbs, medicoes_header, data_medicoes


# data medicoes
print(medicoes_header)
pos = medicoes_header.index('PLreal')

X = np.delete(data_medicoes, pos, 1)
Y = data_medicoes[:, pos]

print("X:")
print (X)
print("\n\n\nY:")
print(Y)
