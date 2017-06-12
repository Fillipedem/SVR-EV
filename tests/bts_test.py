"""
test SVR with cuckoo search for one BTS
"""
# matplotlib and sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# svr and bts
from svr import SVRClassifier
from bts import bts_base, pathloss



# base
X = bts_base[1.0]
y = pathloss[1]['plreal']

# SVRClassifir
svr_classifier = SVRClassifier(X, y)
svr_classifier.set_parameters(13.615709853775128, 0.1, 0.00657299504707376)

# predict
svr_y = svr_classifier.predict(10)

# get mean squared error
print("RMSE: {}".format(mean_squared_error(y, svr_y)))

# plot
plt.scatter(y, svr_y, color='darkorange', label='data')
plt.show()