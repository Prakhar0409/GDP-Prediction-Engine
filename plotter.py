import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('6without_gdp_without_m1m3.csv', delimiter=',')

x = data[:,0]
data = data[:,1:3]

plt.plot(x,data[:,0],label='Predictions')
plt.plot(x,data[:,1],label='Actual')
plt.xlabel('Year')
plt.ylabel('GDP (in crore Rs.)')
#plt.title('Prediction evaluation 4')
plt.legend()
plt.show()
