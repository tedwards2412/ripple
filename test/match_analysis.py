import numpy as np
import matplotlib.pyplot as plt

match_data = np.loadtxt("./ripple_phenomPv2_matches.txt")
match_data[:,0] = match_data[:,0]/(2* 10**30)
match_data[:,1] = match_data[:,1]/(2* 10**30)
for i in range(len(match_data[:,-1])):
    if match_data[i, -1] < 0.3:
        print(match_data[i])
print("###########################")
print("now the healthy ones: ")
print("###########################")
for i in range(len(match_data[:,-1])):
    if match_data[i, -1] >0.999999:
        print(match_data[i])