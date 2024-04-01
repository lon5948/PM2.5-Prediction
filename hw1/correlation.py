import numpy as np
from sys import argv

FEATURE_NUM = 18

train_data = [[] for _ in range(FEATURE_NUM)]
with open(argv[1], 'rb') as f:
    data = f.read().splitlines()
    for index, line in enumerate(data[1:]): # discard the first row - header
        line = [x.replace("\'", "").strip() for x in str(line).split(',')[3:]]
        line = [x.replace("#", "0") for x in line]
        line = [x.replace("*", "0") for x in line]
        line = [x.replace("x", "0") for x in line]
        line = [x.replace("A", "0") for x in line]
        line = [float(x) for x in line]
        train_data[index % 18] += line

np.set_printoptions(precision = 4, suppress = True)
corr = np.corrcoef(train_data[9], [train_data[i] for i in range(FEATURE_NUM)])
for i in range(1, len(corr[0])):
    print(f"Feature {i-1}: {abs(corr[0][i])}")