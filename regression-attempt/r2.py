import sys
from sklearn.metrics import r2_score
import random
#import statistics
import numpy as np

num = 10
y_pred# = random.sample(range(num),num)
y_test# = random.sample(range(num),num)

r2 = r2_score(y_pred,y_test)
print "R2:",r2

y_std = [y_pred[i] - y_test[i] for i in range(len(y_test))]
y_std = np.array(y_std)
stdev = np.std(y_std)

print "Std. Deviation of error:",stdev


