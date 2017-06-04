"""
Import the database from the csv files
"""
import numpy as np

# files
erbs = 'dados_ERBs.csv'
medicoes = 'medicoes.csv'

# dicts
lon_dict = {b'GSM1800': 1.0,
            b'GSM1801': 2.0}

bcch_dict = {b'DX-1710-2170-65-18i-M': 1.0}

erb_dict = {b'"PE136Y"': 1.0,
            b'"PE137X"': 2.0,
            b'"PE137Z"': 3.0,
            b'"PE197X"': 4.0,
            b'PE136Y': 1.0,
            b'PE137X': 2.0,
            b'PE137Z': 3.0,
            b'PE197X': 4.0}

# reading med
med_convert = {11: lambda x: erb_dict[x],}

file = open(medicoes, 'r')
header = file.readline()

medicoes_header = [att for att in header.split('"') if len(att) > 1]
data_medicoes = np.loadtxt(file, delimiter=',', converters=med_convert)

file.close()

# reading erbs
erbs_convert = {0: lambda x: erb_dict[x],
                5: lambda x: lon_dict[x],
                7: lambda x: bcch_dict[x]}

file = open(erbs, 'r')
header = file.readline()

erbs_header = header.split(';')
data_erbs = np.loadtxt(file, delimiter=';', converters=erbs_convert)

file.close()