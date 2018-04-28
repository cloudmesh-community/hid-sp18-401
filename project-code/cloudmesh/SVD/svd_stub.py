####################################################################
#This is Python code for building Deep Neural Network for the MNIST#
# Data , Applying Singular Value Decompostion, Discarding the less #
# significant Singular values and training the model again.        #
#             ########################                             #
# Authors: Priyadarshini Vijjigiri, Goutham Arra                   #
# Time   : Spring-2018                                             # 
####################################################################

#import statements
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
import numpy as np

#Downloading data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

D_dict = dict()
D_dict[10] = 0
D_dict[20] = 1
D_dict[50] = 2
D_dict[100] = 3
D_dict[200] = 4

tf.set_random_seed(4422)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.33
sess = tf.InteractiveSession(config = config)

#Defining the weight matrix, bias vector and activation function
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.truncated_normal([784,1024], stddev=0.1))
b1 = tf.Variable(tf.constant(0.001, shape = [1024]))
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([1024,1024], stddev=0.1))
b2 = tf.Variable(tf.constant(0.001, shape = [1024]))
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)                

W3 = tf.Variable(tf.truncated_normal([1024,1024], stddev=0.1))
b3 = tf.Variable(tf.constant(0.001, shape = [1024]))
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)                 

W4 = tf.Variable(tf.truncated_normal([1024,1024], stddev=0.1))
b4 = tf.Variable(tf.constant(0.001, shape = [1024]))
y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)                  

W5 = tf.Variable(tf.truncated_normal([1024,1024], stddev=0.1))
b5 = tf.Variable(tf.constant(0.001, shape = [1024]))
y5 = tf.nn.relu(tf.matmul(y4, W5) + b5)                 

W6 = tf.Variable(tf.truncated_normal([1024,10], stddev=0.1))
b6 = tf.Variable(tf.constant(0.001, shape = [10]))
y  = tf.matmul(y5, W6) + b6                 
y_pred = tf.nn.softmax(y)               

# cross entropy formula to assess the built model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Back propagation of error is done using RMSPropOptimizer
train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#accuracy is calcuated from true values to predicted values
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Session started here and model is trained after
sess.run(tf.global_variables_initializer())

for i in range(1000):
      batch = mnist.train.next_batch(256)

      train_step.run(feed_dict={x: batch[0], y_: batch[1]})                 
    
original_test_accuracy = str(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})) 

############# Saving the weight matrices and bias vectors obtained by training with 5 hidden layers.###############

W = [W1,W2,W3,W4,W5]
b = [b1,b2,b3,b4,b5]

####################################################################################################################
########Applying Singular Value Decomposition on these weight matrices and reducing the complexity##################

S_array = []
U_array = []
V_array = []
for i in range(0,5):
    S,U,V = tf.svd(W[i])
    S_hat=[]
    U_hat=[]
    V_hat=[]
    for j in [10,20,50,100,200,S.get_shape().as_list()[0]] :
      S_hat.append(tf.diag(S[0:j]))
      U_hat.append(U[:,:j])
      V_hat.append(V[:,:j]) 
    S_array.append(S_hat)
    U_array.append(U_hat)
    V_array.append(V_hat)


####################################################################################################################
########Training model again using reduced weight matrices for different values of D. i,e.10,20,50,100,200 #########




Accuracies = []
for D in [10,20,50,100,200]:
  
  tf.set_random_seed(4422)
  x_d = tf.placeholder(tf.float32, shape=[None, 784])
  y_d = tf.placeholder(tf.float32, shape=[None, 10])

  U_d = []
  V_d = []
  S_d = []
  for i in range(0,5):
    #list of reduced singular matrices
    U_d.append(tf.Variable(U_array[i][D_dict[D]]))
    S_d.append(tf.Variable(S_array[i][D_dict[D]]))
    V_d.append(tf.Variable(tf.transpose(V_array[i][D_dict[D]])))

  #each weight matrix has factorized into U,S,V and after applying SVD S and V are multiplied and combinely called V. 
  # Five complex layers with 6 weight matrices is converted to 10 simple layers with 12 weight matrices.
  
  y11_d = tf.matmul(x_d,U_d[0])
  y12_d = tf.nn.relu(tf.matmul(y11_d, tf.matmul(S_d[0],V_d[0])) + b1)

  y21_d = tf.matmul(y12_d,U_d[1])
  y22_d = tf.nn.relu(tf.matmul(y21_d, tf.matmul(S_d[1],V_d[1])) + b2)

  y31_d = tf.matmul(y22_d,U_d[2])
  y32_d = tf.nn.relu(tf.matmul(y31_d, tf.matmul(S_d[2],V_d[2])) + b3)

  y41_d = tf.matmul(y32_d,U_d[3])
  y42_d = tf.nn.relu(tf.matmul(y41_d, tf.matmul(S_d[3],V_d[3])) + b4)

  y51_d = tf.matmul(y42_d,U_d[4])
  y52_d = tf.nn.relu(tf.matmul(y51_d, tf.matmul(S_d[4],V_d[4])) + b5)

  W6_d = W6 
  y6_d = tf.matmul(y52_d,W6_d) + b6


  cross_entropy_d = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_d, logits=y6_d))

  train_step_d = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy_d)

  correct_prediction_d = tf.equal(tf.argmax(y6_d,1), tf.argmax(y_d,1))

  accuracy_d = tf.reduce_mean(tf.cast(correct_prediction_d, tf.float32))

  sess.run(tf.global_variables_initializer())


  for i in range(1000):
    batch = mnist.train.next_batch(256)

    train_step_d.run(feed_dict={x_d: batch[0], y_d: batch[1]})
  # accuracy values for different D's are stored in a list
  svd_test_accuracy = accuracy_d.eval(feed_dict={x_d: mnist.test.images, y_d: mnist.test.labels})
  Accuracies.append(svd_test_accuracy)

def svd_main():
       
    ans = "The Original MNIST Network test accuracy is " + str(original_test_accuracy) + " After doing SVD on the original network with D value as 10, we get a test accuracy of "  + str(Accuracies[0]) + " With D value as 20, we get a test accuracy of "  + str(Accuracies[1]) + " With D value as 50, we get a test accuracy of "  + str(Accuracies[2]) + " With D value as 100, we get a test accuracy of "  + str(Accuracies[3]) + " With D value as 200, we get a test accuracy of "  + str(Accuracies[4]) 
    return ans

