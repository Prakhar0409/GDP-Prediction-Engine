import numpy as np
import scipy
from sklearn.metrics import r2_score

# Read and shuffle data
data = np.genfromtxt('data.csv', delimiter=',')
m = data.shape[0]

years = data[:,-1]
data = scipy.delete(data,-10,1)      #remove old gdp
#data = scipy.delete(data,-9,1)
#data = scipy.delete(data,-8,1)
#data = scipy.delete(data,-7,1)
#data = scipy.delete(data,-6,1)
#data = scipy.delete(data,-5,1)

data = scipy.delete(data,-1,1)      #remove year column
#data = scipy.delete(data,11,1)      #remove imports
#data = scipy.delete(data,10,1)      #remove exports
data = scipy.delete(data,8,1)      #remove m3
data = scipy.delete(data,7,1)      #remove m1
data = scipy.delete(data,2,1)      #remove population count

validate = np.empty_like(data)
validate[:] = data
np.random.shuffle(data)
#divide data in train vs validation
#validate = data[m/2:,:]
data = data[:m/2,:]

#training phase

## Model params
m = data.shape[0]
mu = data.sum(axis=0)/m
x_minus_mu = data - mu
sigma = np.dot(x_minus_mu.T ,x_minus_mu)/m

y1 = data[:,0]
y2 = data[:,1:]
lmu1 = mu[0]
lmu2 = mu[1:]

lsigma11 = sigma[0,0]
lsigma12 = sigma[0,1:]
lsigma21 = sigma[1:,0]
lsigma22 = sigma[1:,1:]

# predicting on validation set
#validate = np.genfromtxt('data.csv', delimiter=',')
#validate = scipy.delete(validate,-1,1)
#validate = scipy.delete(validate,-9,1)      #remove old gdp
m = validate.shape[0]
predict = np.zeros(m)

y1 = validate[:,0]
y2 = validate[:,1:]
tmp = y2 - lmu2
print(tmp.shape)

for i in range(m):
    predict[i] = lmu1 + np.dot(np.dot(lsigma12,np.linalg.pinv(lsigma22)), tmp[i].T)

print("####### PREDICTIONS #######")
print(predict.shape)
for i in range(m):
    print(str(years[i])+"     "+str(predict[i])+"   "+str(y1[i])+"     "+str(predict[i]/y1[i])+"    "+str(predict[i]-y1[i]))

print("#### Other Stats #####")
r2 = r2_score(predict,y1)
print("R2:",r2)

y_std = [predict[i] - y1[i] for i in range(len(y1))]
y_std = np.array(y_std)
stdev = np.std(y_std)
print("Std. Deviation of error:",stdev)

tmp = abs(predict-y1)
avg = tmp.sum(axis=0)/m
print("Mean Absolute Error:",avg)
