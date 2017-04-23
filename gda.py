import numpy as np
import scipy

# Read and shuffle data
data = np.genfromtxt('data.csv', delimiter=',')
np.random.shuffle(data)

##data = data[:,1:]                 #remove year column
#data = scipy.delete(data,4,1)      #remove disaster
#data = scipy.delete(data,2,1)      #remove population count

#divide data in train vs validation
m = data.shape[0]
validate = data[:m/4,:]
data = data[m/4:,:]

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
    print(str(predict[i])+"   "+str(y1[i])+"     "+str(predict[i]/y1[i]))

predict = abs(predict-y1)
avg = predict.sum(axis=0)/m
print(avg)
