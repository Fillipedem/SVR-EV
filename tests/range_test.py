"""
Try to search best values for parameters in pre defined range
"""
from fitness import fitness
from sklearn.metrics import mean_squared_error
from bts import bts_base, pathloss

# BTS
erb = 4

# first use the paper's approach
range_C = 45
range_epsilon = 1.0
div = 18

"""
# first get C
results = []
c_value = range_C/div


for i in range(1, div):
    C = c_value*i

    rmse = fitness([C, 1])

    results.append(rmse)

# get best parameter C
best_c = results.index(min(results))
C = (best_c + 1)*c_value


# change gamma
results = []
epsilon_value = range_epsilon/div

for i in range(1, div):
    ep = epsilon_value * i
    
    rmse = fitness([C, ep])

    results.append(rmse)

# get get parameter C and RMSE
rmse = min(results)
idx = results.index(rmse)
ep = epsilon_value*(idx + 1)

print("Papers:")
print("RMSE: {}, for C: {}, epsilon: {}".format(rmse, C, ep))
print()
"""

# other models
y = pathloss[erb]['plreal']
y_predict = pathloss[erb]['plecc33']

# ECC-33
rmse = mean_squared_error(y, y_predict)

print("ECC-33:")
print("RMSE: {}".format(rmse))
print()


# PHLata
y_predict = pathloss[erb]['plhata']
rmse = mean_squared_error(y, y_predict)

print("PHLata:")
print("RMSE: {}".format(rmse))
print()

