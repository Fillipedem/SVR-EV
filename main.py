"""
Reprodução dos Testes e plot dos graficos do artigo
"""
import numpy as np
import matplotlib.pyplot as plt
from bts import bts_base, bts_header, pathloss
from sklearn.svm import SVR


# SVR
svr_rbf = SVR(kernel='rbf', C=16)

svr_rbf.fit(bts_base[2][:1500], pathloss[2]['plreal'][:1500])

y = svr_rbf.predict(bts_base[2][1500:])
#y = pathloss[2]['plecc33'][1500:]
plreal = pathloss[2]['plreal'][1500:]


# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((y - plreal) ** 2))
# Explained variance score: 1 is perfect prediction
print('SVR score: %.2f' % svr_rbf.score(bts_base[2][1500:], plreal))

# Plot
lw = 2
plt.scatter(plreal, y, color='darkorange', label='data')
plt.show()
