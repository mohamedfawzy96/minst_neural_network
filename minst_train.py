import tensorflow as tf
from neural_network import NN
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = tf.one_hot(y_train, depth=10).numpy()
y_test = tf.one_hot(y_test, depth=10).numpy()

x_train = x_train.reshape(x_train.shape[0], -1).T
y_train = y_train.reshape(y_train.shape[0], -1).T
dev_X = x_test.reshape(x_test.shape[0], -1).T
dev_Y = y_test.reshape(y_test.shape[0], -1).T
nn = NN([784, 10, 40, 20, 10])
x = tf.Variable(x_train, dtype='float32')
y = tf.Variable(y_train, dtype='float32')
nn.model(x, y, dev_X, dev_Y, num_epochs=100, learning_rate=0.001, minibatch_size=512, beta_1=0.9, lamda=0,
         softmax=True)