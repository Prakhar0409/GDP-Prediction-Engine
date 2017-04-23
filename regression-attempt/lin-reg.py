import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import io
import csv
import random
import sys

def make_data(test_ratio=0.1):
    features = []
    strr = sys.argv[1]
    with io.open(strr, newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rownum = 0
        for row in digitreader:
            if rownum == 0:
                rownum += 1
                continue
#             row = [float(x) for x in row]
#             str_list = row[0].split(',')
            #print row
            flt_list = [float(x.replace(',', '')) for x in row]
            
#             y_list = np.zeros([10])
#             y_list[int_list[0]] = 1
            lenn = 25
            if len(strr)<9:
                lenn = 13
            y_list = list([flt_list[lenn]])
            features.append([flt_list[1:lenn], y_list])
            rownum += 1

    random.shuffle(features)
    features = np.array(features)
    print(len(features))
    
    test_size = int(len(features) * test_ratio)
    x_train = list(features[:,0][:-test_size])
    y_train = list(features[:,1][:-test_size])
    x_test = list(features[:,0][-test_size:])
    y_test = list(features[:,1][-test_size:])
    #x_test = x_train + x_test
    #y_test = y_train + y_test
    return x_train, y_train, x_test, y_test
    

x_train, y_train, x_test, y_test = make_data()
rep = int(sys.argv[2])
print "Rep:",rep
x_train, y_train = rep*x_train, rep*y_train#, rep*x_test, rep*y_test
#sample_weight = [1]*12
#sample_weight[11] = 9

regr = linear_model.LinearRegression(normalize=True)
regr.fit(x_train,y_train)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))

y_pred = regr.predict(x_test)

r2 = r2_score(y_pred,y_test)
print "R2:",r2

y_std = [y_pred[i] - y_test[i] for i in range(len(y_test))]
y_std = np.array(y_std)
stdev = np.std(y_std)

print "Std. Deviation of error:",stdev

print y_test/y_pred


#clf = SVR(C=1.0, epsilon=0.2)
#print "Initialized"
#clf.fit(x_train, y_train)
#print "Trained"
#y_predict = clf.predict(x_test)
#print y_predict[0]
#r_squared = clf.score(x_test,y_test)
#print r_squared
