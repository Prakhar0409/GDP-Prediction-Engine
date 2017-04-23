import numpy as np
import sys

data = np.genfromtxt('data1.csv', delimiter=',')
print(np.corrcoef(data[1:,-1],data[1:,-2]))

m = data.shape[0]
gdp = data[:,0]
for i in range(data.shape[1]):
    monsoon = data[:,i]
    print(np.corrcoef(gdp,monsoon)[0,1])

avg_mon = monsoon.sum(axis=0)/m
avg_gdp = gdp.sum(axis=0)/m

monsoon = monsoon - avg_mon
gdp = gdp - avg_gdp

norm_monsoon = np.linalg.norm(monsoon)
norm_gdp = np.linalg.norm(gdp)

monsoon = monsoon/norm_monsoon
gdp = gdp/norm_gdp

res = np.dot(monsoon,gdp)

print(res)
