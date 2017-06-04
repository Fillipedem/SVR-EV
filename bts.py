"""
Separa o banco de dados para cada BTS(Base Transceiver Station )
"""
from database import medicoes_header, data_medicoes
import numpy as np

# Training base for each BTS
bts_base = {1.0: [],
       2.0: [],
       3.0: [],
       4.0: []}

# new bts header
bts_header = []

# PLhata path loss
plhata = []

# PLecc33 path loss
plecc33 = []

# readl path loss
plreal = []



#              #
# Main program #
#              #

# find ERB type column position
pos = medicoes_header.index('Erb')

# separado base para cada BTS
for bts in bts_base:
    bts_base[bts] = list(filter(lambda m: m[pos] == bts, data_medicoes))

print(bts_base[1])
# now prepare the database
remove_colum = []
remove_colum.append('Erb')
remove_colum.append('PLecc33')
remove_colum.append('PLreal')

# new hedaer
bts_header = medicoes_header[:]
    
# remove data that will not be used for the test
for col_name in remove_colum:
    index = bts_header.index(col_name)

    # update bts_header
    bts_header = bts_header[:index] + bts_header[index + 1:]
    
    # update bts bases
    for bts in bts_base:
        base = bts_base[bts]
        base = np.delete(base, index, 1)
    

# setting up PLecc33 and PLhata data to be compared
index = medicoes_header.index('PLecc33')
plecc33 = data_medicoes[:, index]

index = medicoes_header.index('PLhata')
plhata = data_medicoes[:, index]

# getting the real Path loss
index = medicoes_header.index('PLreal')
plreal = data_medicoes[:, index]