"""
Reprodução dos Testes e plot dos graficos do artigo
"""
import numpy as np
import matplotlib.pyplot as plt
from bts import bts_base, bts_header, pathloss
from sklearn.svm import SVR


# SVR
svr_rbf = SVR(kernel='rbf', C=16)

svr_rbf.fit(bts_base[1][:1500], pathloss[1]['plreal'][:1500])

y = svr_rbf.predict(bts_base[1][1500:])
plreal = pathloss[1]['plreal'][1500:]

# Plot
lw = 2
plt.scatter(plreal, y, color='darkorange', label='data')
plt.show()
