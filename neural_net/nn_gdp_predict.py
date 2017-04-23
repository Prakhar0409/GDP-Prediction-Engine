
# coding: utf-8

# In[3]:

import tensorflow as tf
import csv
import random
import numpy as np


# In[4]:

l1_nodes = 30
l2_nodes = 60
n_classes = 1
hm_epochs = 200
batch_size = 10


# In[5]:

x = tf.placeholder('float')
y = tf.placeholder('float')


# In[7]:

def make_data(test_ratio=0.1):
    features = []
    with open('data.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rownum = 0
        for row in digitreader:
            if rownum == 0:
                rownum += 1
                continue
#             row = [float(x) for x in row]
#             str_list = row[0].split(',')
            flt_list = [float(x.replace(',', '')) for x in row]
            
        
            y_list = list([flt_list[12]])
            features.append([flt_list[1:12], y_list])
            rownum += 1

    random.shuffle(features)
    features = np.array(features)
#     print(features)
    
#     x_norm = normalize(np.array(features[:,0]), norm='l2', axis=0, copy=True, return_norm=False)
    
    test_size = int(len(features) * test_ratio)
    x_train = list(features[:,0][:-test_size])
    y_train = list(features[:,1][:-test_size])
    x_test = list(features[:,0][-test_size:])
    y_test = list(features[:,1][-test_size:])
    return x_train, y_train, x_test, y_test
    

x_train, y_train, x_test, y_test = make_data()
print(y_test[0])
print(x_test[0])


# In[11]:

def neural_net_model(x):
    num_x = 11
    num_cl = n_classes
    l1_hidden = {'weight':tf.Variable(tf.random_normal([num_x, l1_nodes])),
                'bias':tf.Variable(tf.random_normal([l1_nodes]))}
    
#     l2_hidden = {'weight':tf.Variable(tf.zeros([l1_nodes, l2_nodes])),
#                 'bias':tf.Variable(tf.zeros([l2_nodes]))}
    
    ltp_hidden = {'weight':tf.Variable(tf.zeros([num_x,num_cl])),
                'bias':tf.Variable(tf.zeros([num_cl]))}
    
    l_output = {'weight':tf.Variable(tf.zeros([l1_nodes, n_classes])),
                'bias':tf.Variable(tf.zeros([n_classes]))}
    
    l1 = tf.add(tf.matmul(x, l1_hidden['weight']), l1_hidden['bias'])
    l1 = tf.nn.sigmoid(l1)
    
#     l2 = tf.add(tf.matmul(l1, l2_hidden['weight']), l2_hidden['bias'])
#     l2 = tf.nn.relu(l2)
    
    
#     output = tf.add(tf.matmul(l1, l_output['weight']), l_output['bias'])
#     output = tf.nn.softmax(output)
    output = tf.add(tf.matmul(x, ltp_hidden['weight']), ltp_hidden['bias'])
#     output = tf.nn.softmax(output)
    
    return output


# In[12]:

def train_network():
    prediction = neural_net_model(x)
    cost = tf.reduce_mean(tf.square(prediction-y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
#             random.shuffle(x_train)
            i = 0
#             tf.nn.l2_normalize(x_train, dim, epsilon=1e-12, name=None)
            while(i<len(x_train)):
                start = i
                end = i + batch_size
                batch_x = x_train[start:end]
                batch_y = y_train[start:end]
                
                c,_ = sess.run([cost, optimizer],feed_dict={x:batch_x, y:batch_y[0]})
#                 print("c",c)
                
                epoch_loss += c
                i += batch_size
                
            correct = tf.sqrt(tf.square(prediction - y))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))

            if epoch%10 == 0:
                print("train accuracy:",accuracy.eval({x:x_train,y:y_train}))

                print("In epoch",epoch,"of",hm_epochs,"epochs, loss=",epoch_loss)
            
        correct = tf.sqrt(tf.square(prediction - y))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print("accuracy:",accuracy.eval({x:x_test,y:y_test}), " yo: ", prediction.eval({x:x_test})/y_test)
    
train_network()



# In[ ]:



