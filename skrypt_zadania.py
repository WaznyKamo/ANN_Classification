# Klasyfikacja punktow(obrazow) nalezacych do prostokata
#
#       (1,3) o----------------o (4,3)
#             |                |
#             |                |
#             |                |
#       (1,1) o----------------o (4,1)
#

import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
# print(tf.__version__)
tf.compat.v1.disable_eager_execution()


points = np.array([[1, 1], [1, 3], [4, 3], [4, 1], [2, 2], [5, 2], [5, 3], [4, 4], [1, 4], [3, 5], [0.5, 2],
                   [2, 3], [3, 3], [2, 1], [4, 2], [2, 3], [4, 5], [4, 3], [0.5, 0.5], [3, 4], [5, 1], [2, 5], [2, 4], [5, 0], [2, 0], [3, 0.5]], dtype=float)
class_labels = np.transpose(np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float))

ind0 = np.where(class_labels == 0)       # indexes of examples which belong to class 0
ind1 = np.where(class_labels == 1)

nn_1st = [20]                     # number of neurons in the 1-st level of ANN
nn_2nd = [16]                   # number of neurons in the 2-nd level of ANN

num_of_features = 2
num_of_epochs = 4500
num_to_show = 500

sess = tf.compat.v1.InteractiveSession()

x = tf.compat.v1.placeholder(tf.float32, shape=[None, num_of_features])  # place for input vectors
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])               # place for desired output of ANN

W1 = tf.Variable(tf.random.truncated_normal([num_of_features, nn_1st[0]], stddev=0.1))   # 1-st level weights - trainable version
b1 = tf.Variable(tf.constant(0.1, shape=[nn_1st[0]]))    # 1-st level biases -||-

h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)                                    # output values from 1-st level (using hyperbolic tangent activation func.)

W2 = tf.Variable(tf.random.truncated_normal([nn_1st[0], nn_2nd[0]], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[nn_2nd[0]]))

h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)                                    # output values from 1-st level (using hyperbolic tangent activation func.)


W_last = tf.Variable(tf.random.truncated_normal([nn_2nd[0], 1], stddev=0.2))               # 2-nd level weights values
b_last = tf.Variable(tf.zeros([1]))                                           # 2-nd level bias values

y = tf.nn.sigmoid(tf.matmul(h2, W_last) + b_last)                   # output from ANN (single value using sigmoidal act.funct in range (0,1))


weights_analytic = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
my_tensor = tf.constant(weights_analytic, dtype=float)
W1_a = tf.Variable(my_tensor)                                # 1-st level weights - analytical version
b1_a = tf.Variable([-1, 4, -1, 3])    # 1-st level biases -||-

h1_a = tf.round(tf.nn.sigmoid(tf.matmul(tf.cast(x, tf.float32), tf.cast(W1_a, tf.float32)) + tf.cast(b1_a, tf.float32)))                    # analytical version

W_last_a = tf.Variable([[1], [1], [1], [1]])
b_last_a = tf.Variable(-3.5)                                           # 2-nd level bias values

y_a = tf.round(tf.nn.sigmoid(tf.matmul(tf.cast(h1_a, tf.float32), tf.cast(W_last_a, tf.float32)) + tf.cast(b_last_a, tf.float32)))           # output from ANN - analytical version


mean_square_error = tf.reduce_mean(tf.reduce_sum((y_ - y)*(y_ - y)))          # MSE loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(y) + y*tf.math.log(y_+0.001)))  # full cross-entropy loss function

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(mean_square_error)   # training method, step value, loss function

init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()
sess.run(init)

# the training process:
for epoch in range(num_of_epochs+1):
    sess.run(train_step, feed_dict={x: points, y_: class_labels})     # ses.run using dictionary with whole training data
    if epoch % num_to_show == 0:
        wrong_prediction = tf.greater(tf.abs(y-y_), 0.6)               # vector of classification errors
        error = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))  # mean classification error

        print("\nafter "+str(epoch)+" epoch")
        print("MSE error = " + str(sess.run(mean_square_error, feed_dict={x: points, y_: class_labels})))
        print("Cross Entropy error = " + str(sess.run(cross_entropy, feed_dict={x: points, y_: class_labels})))
        print("training classification error = " + str(sess.run(error, feed_dict={x: points, y_: class_labels})))


# drawing points:
line1 = plt.plot(np.transpose(points[ind0[0], 0]), np.transpose(points[ind0[0], 1]), 'ro', label='class 0')
line2 = plt.plot(np.transpose(points[ind1[0], 0]), np.transpose(points[ind1[0], 1]), 'bs', label='class 1')
plt.title("points in 2d plane - decision boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

# drawing decision boundary:
X1, X2 = np.meshgrid(np.linspace(0, 8, 120), np.linspace(0,8, 120))  # grid of points in 2D plane
P = np.stack((X1.flatten(),X2.flatten()), axis=1)                    # points formated for ANN input
Y = sess.run(y_a, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y,X1.shape)                                           # reshaping to shape of grid
plt.contourf(X1, X2, Z, levels=[0.5, 1.0])                           # curve for level=0.5 - a decision boundary, shaded class 1 area
plt.title('analytical method')
plt.show()

# drawing 3D mesh:
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.viridis)
plt.title('analytical method')
plt.show()

# drawing points:
line1 = plt.plot(np.transpose(points[ind0[0], 0]), np.transpose(points[ind0[0], 1]), 'ro', label='class 0')
line2 = plt.plot(np.transpose(points[ind1[0], 0]), np.transpose(points[ind1[0], 1]), 'bs', label='class 1')
plt.title("points in 2d plane - decision boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

# drawing decision boundary:
X1, X2 = np.meshgrid(np.linspace(0, 8, 120), np.linspace(0, 8, 120))  # grid of points in 2D plane
P = np.stack((X1.flatten(), X2.flatten()), axis=1)                    # points formatted for ANN input
Y = sess.run(y, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y, X1.shape)                                           # reshaping to shape of grid
plt.contourf(X1, X2, Z, levels=[0.5, 1.0])                           # curve for level=0.5 - a decision boundary, shaded class 1 area
plt.title('learning method')
plt.show()

# drawing 3D mesh:
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
Y = sess.run(y, feed_dict={x: P})                                    # ANN outputs for flatten grid  points
Z = np.reshape(Y, X1.shape)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap=cm.viridis)
plt.title('learning method')
plt.show()
