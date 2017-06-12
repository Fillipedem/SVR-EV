"""
Reprodução dos Testes e plot dos graficos do artigo
"""
import numpy as np
import matplotlib.pyplot as plt
from bts import bts_base, bts_header, pathloss
from sklearn.svm import SVR

erb = 4

# SVR
svr_rbf = SVR(kernel='rbf', C=44.424063740, epsilon=0.0056846371, gamma=1/2000)

svr_rbf.fit(bts_base[erb][:1500], pathloss[erb]['plreal'][:1500])

y = svr_rbf.predict(bts_base[erb][1500:])
plreal = pathloss[erb]['plreal'][1500:]


# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((y - plreal) ** 2))
# Explained variance score: 1 is perfect prediction
print('SVR score: %.2f' % svr_rbf.score(bts_base[erb][1500:], y))

# Plot
lw = 2
plt.plot([plreal.min(), plreal.max()], [plreal.min(), plreal.max()], lw=2, color="black")
plt.scatter(plreal, y, color='darkorange', label='data', edgecolors="black")
plt.xlabel("Predict Path Loss")
plt.ylabel("Real Path Loss")
plt.show()
