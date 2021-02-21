import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class NN:
    weights = []
    biases = []

    def __init__(self, layers):
        self.layers = layers
        self.init_weights()

    def init_weights(self):
        self.weights = [tf.Variable(tf.initializers.GlorotNormal()(shape=[self.layers[i], self.layers[i - 1]])) for i
                        in
                        range(1, len(self.layers))]
        self.biases = [tf.Variable(tf.initializers.GlorotNormal()(shape=[self.layers[i], 1])) for i in
                       range(1, len(self.layers))]

    def forward_prop(self, X):
        Z = tf.add(tf.matmul(self.weights[0], X), self.biases[0])
        prev_A = tf.nn.relu(Z)
        for i in range(1, len(self.weights)):
            Z = tf.add(tf.matmul(self.weights[i], prev_A), self.biases[i])
            prev_A = tf.nn.relu(Z)
        return Z

    def compute_cost(self, Z3, Y, m, lmda, softmax=False):
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        cost_l2 = (lmda / m) * tf.math.reduce_sum(
            [tf.math.reduce_sum(tf.math.square(weight)) for weight in self.weights])

        if softmax:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) + cost_l2
            return cost

        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) + cost_l2

    def model(self, X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
              num_epochs=1500, minibatch_size=1000, print_cost=True, beta_1=0.9, lamda=0.001, softmax=False):
        (n_y, _) = Y_train.shape
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
        costs = []
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            mini_batches, num_minibatches = self.random_mini_batch(X_train, Y_train, minibatch_size)
            print("no minibatch:", len(mini_batches))
            for (X_mini, Y_mini) in mini_batches:
                with tf.GradientTape() as tape:
                    Z3 = self.forward_prop(X_mini)
                    loss_value = self.compute_cost(Z3, Y_mini, lmda=lamda, m=X_mini.shape[1], softmax=softmax)

                gradients = tape.gradient(loss_value, self.weights + self.biases)
                optimizer.apply_gradients(zip(gradients, self.weights + self.biases))
                epoch_cost += loss_value / num_minibatches
                costs.append(epoch_cost)

            acc = self.accuracy(X_train, Y_train, many_classes=softmax)
            test_acc = self.accuracy(X_test, Y_test, many_classes=softmax)
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("train acc after epoch %i: %f" % (epoch, acc))
            print("test acc after epoch %i: %f" % (epoch, test_acc))
            plt.show()
            print("#########")

        plt.plot(costs, 'r')
        plt.show()

    def random_mini_batch(self, X, Y, mini_batch_size):
        m = X.shape[1]  # number of training examples
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = tf.gather(X, axis=1, indices=permutation)
        shuffled_Y = tf.gather(Y, axis=1, indices=permutation)
        num_complete_minibatches = math.floor(m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size:(k * mini_batch_size) + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k * mini_batch_size) + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            final_batch_size = m - (mini_batch_size * (int(m / mini_batch_size)))
            mini_batch_X = shuffled_X[:, -final_batch_size:]
            mini_batch_Y = shuffled_Y[:, -final_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
            num_complete_minibatches = num_complete_minibatches + 1
        return mini_batches, num_complete_minibatches

    def accuracy(self, X, Y, many_classes=False):
        forward = self.forward_prop(X)

        if many_classes:
            correct_prediction = tf.equal(tf.argmax(forward), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        else:
            sigmoid_vals = tf.sigmoid(forward)
            sigmoid_vals = sigmoid_vals > 0.5
            Y = Y == 1.
            accuracy = np.mean(Y == sigmoid_vals)
        return accuracy

