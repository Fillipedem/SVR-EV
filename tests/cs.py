"""
Running Cuckoo Search to find best fit
"""

# svr and bts
from fitness import fitness
from optimization.cuckoosearch import CuckooSearch

# Cuckoo Search
cs = CuckooSearch(2, fitness, [(0, 45), (0, 1)], num_nest=15, p=0.25)

# search
[C, epsilon] = cs.search(100)

# print best values
print("Best values are C: {}, and epsilon: {}". format(C, epsilon))





