"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

tf.set_random_seed(1)


class AutoEncoderModel:

    def __init__(self, args):


        # Hyper Parameters
        self.BATCH_SIZE = 64
        self.LR = 0.002         # learning rate
        self.N_TEST_IMG = 5

        # Mnist digits
        self.mnist = input_data.read_data_sets('mnist', one_hot=False)     # use not one-hotted target data
        self.test_x = mnist.test.images[:200]
        self.test_y = mnist.test.labels[:200]

        # plot one example
        print(self.mnist.train.images.shape)     # (55000, 28 * 28)
        print(self.mnist.train.labels.shape)     # (55000, )
        plt.imshow(self.mnist.train.images[0].reshape((28, 28)), cmap='gray')
        plt.title('%i' % np.argmax(self.mnist.train.labels[0]))
        plt.show()

        # tf placeholder
        self.tf_x = tf.placeholder(tf.float32, [None, 28*28])    # value in the range of (0, 1)

        # encoder
        self.en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
        self.en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
        self.en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
        self.encoded = tf.layers.dense(en2, 3)

        # decoder
        self.de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
        self.de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
        self.de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
        self.decoded = tf.layers.dense(de2, 28*28, tf.nn.sigmoid)

        self.loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
        self.train = tf.train.AdamOptimizer(LR).minimize(loss)



    def train(self): #进行训练
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # initialize figure
        f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
        plt.ion()   # continuously plot

        # original data (first row) for viewing
        view_data = self.mnist.test.images[:N_TEST_IMG]
        for i in range(N_TEST_IMG):
            a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
            a[0][i].set_xticks(()); a[0][i].set_yticks(())

        for step in range(8000):
            b_x, b_y = self.mnist.train.next_batch(BATCH_SIZE)
            _, encoded_, decoded_, loss_ = sess.run([self.train, self.encoded, self.decoded, self.loss], {self.tf_x: b_x})

            if step % 100 == 0:     # plotting
                print('train loss: %.4f' % loss_)
                # plotting decoded image (second row)
                decoded_data = sess.run(self.decoded, {self.tf_x: view_data})
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(()); a[1][i].set_yticks(())
                plt.draw(); plt.pause(0.01)
        plt.ioff()

        # visualize in 3D plot
        view_data = self.test_x[:200]
        encoded_data = sess.run(self.encoded, {tf_x: view_data})
        fig = plt.figure(2); ax = Axes3D(fig)
        X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
        for x, y, z, s in zip(X, Y, Z, test_y):
            c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
        plt.show()