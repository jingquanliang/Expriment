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
        self.BATCH_SIZE = args.batch_size
        self.LR = args.rho         # learning rate of endcoder-decoder
        self.N_TEST_IMG = 5
        self.args=args

        '''Placeholder for Learning rate of L1_loss'''
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)

        # Mnist digits
        # self.mnist = input_data.read_data_sets('mnist', one_hot=False)     # use not one-hotted target data
        # self.test_x = self.mnist.test.images[:200]
        # self.test_y = self.mnist.test.labels[:200]

        # plot one example
        # print(self.mnist.train.images.shape)     # (55000, 28 * 28)
        # print(self.mnist.train.labels.shape)     # (55000, )
        # plt.imshow(self.mnist.train.images[0].reshape((28, 28)), cmap='gray')
        # plt.title('%i' % np.argmax(self.mnist.train.labels[0]))
        # plt.show()

        #对输入的数据进行初始化，构建图模型
        self.dataInitalization(args)
        # tf placeholder
        # self.tf_x = tf.placeholder(tf.float32, [None, args.size])    # value in the range of (0, 1)

        # encoder
        #batch_behaviors_weighted_representations在dataInitalization函数中
        self.en0 = tf.layers.dense(self.batch_behaviors_weighted_representations, 128, tf.nn.sigmoid)
        self.en1 = tf.layers.dense(self.en0, 64, tf.nn.sigmoid)
        self.en2 = tf.layers.dense(self.en1, 40, tf.nn.sigmoid)
        self.encoded = tf.layers.dense(self.en2, args.encoderSize,tf.nn.sigmoid)

        # decoder
        self.de0 = tf.layers.dense(self.encoded, 40, tf.nn.sigmoid)
        self.de1 = tf.layers.dense(self.de0, 64, tf.nn.sigmoid)
        self.de2 = tf.layers.dense(self.de1, 128, tf.nn.sigmoid)
        self.decoded = tf.layers.dense(self.de2, args.size, tf.nn.sigmoid)

        self.loss = tf.losses.mean_squared_error(labels=self.batch_behaviors_weighted_representations, predictions=self.decoded)
        self.train = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

        self.sessInitalization(args)#初始化sess


    def sessInitalization(self,args):

        '''Configurations for Tensorflow session'''
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=self.args.threads,
                                     inter_op_parallelism_threads=self.args.threads)

        '''Start Tensorflow session'''
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        print('---' * 30)
        print('Fire up TensorFlow session')


    def dataInitalization(self,args):

        '''Constant for item type weights (non-zero real values)'''
        '''item类型的权重，从文本文件中，我们可以看到，类型就是0，1,2,3四个'''
        self.item_type_weights = tf.constant([1, 1, 1, 1], name='item_type_weights', dtype=tf.float32)
        '''item类型的数量'''
        self.number_of_item_types = tf.cast(tf.count_nonzero(self.item_type_weights), tf.int32)

        '''Variable for context item embeddings'''
        self.target_item_embeddings = tf.get_variable(name='target_item_embeddings',
                                                      shape=[args.number_of_items, args.size],
                                                      initializer=tf.random_uniform_initializer(minval=-0.001,
                                                                                                maxval=0.001),
                                                      dtype=np.float32)

        '''Placeholders for behaviors (observed + negative sampled) in each batch'''
        self.batch_behaviors_item_indices = tf.placeholder(name='batch_behaviors_item_indices',
                                                           shape=[args.batch_size, (args.negative + 1), None],
                                                           dtype=tf.int32)

        '''item类型索引'''
        self.batch_behaviors_item_type_indices = tf.placeholder(name='batch_behaviors_item_type_indices',
                                                                shape=[args.batch_size, (args.negative + 1), None],
                                                                dtype=tf.int32)
        '''行为的标签'''
        self.batch_behaviors_labels = tf.placeholder(name='batch_behaviors_labels',
                                                     shape=[args.batch_size, (args.negative + 1)],
                                                     dtype=tf.float32)

        '''Inflate each item index and item type index into an one_hot vector for each behavior'''
        '''把item和item type转变为index的形式'''
        self.batch_behaviors_item_one_hots = tf.one_hot(name='batch_behaviors_item_one_hots',
                                                        indices=self.batch_behaviors_item_indices,
                                                        depth=args.number_of_items)

        self.batch_behaviors_item_type_one_hots = tf.one_hot(name='batch_behaviors_item_type_one_hots',
                                                             indices=self.batch_behaviors_item_type_indices,
                                                             depth=self.number_of_item_types)

        '''Assemble itemset embeddings representation for each behavior，矩阵的运算，把每一个item根据embeddings矩阵转变'''
        self.batch_behaviors_item_embeddings = tf.einsum('ijth,hd->ijtd',
                                                         self.batch_behaviors_item_one_hots,
                                                         self.target_item_embeddings)

        '''Match item weights for each behavior，外积，对应元素相乘即可，其实，就是把type和type的weight相乘'''
        self.batch_behaviors_item_weights = tf.einsum('ijth,h->ijt',
                                                      self.batch_behaviors_item_type_one_hots,
                                                      self.item_type_weights)

        '''Weight behavior itemset embeddings representation，这个就是behavior真正的表示'''
        self.batch_behaviors_weighted_representations = tf.einsum('ijtd,ijt->ijtd',
                                                                  self.batch_behaviors_item_embeddings,
                                                                  self.batch_behaviors_item_weights)

    def getTarget_embeddings(self):

        '''Write out target embeddings'''
        target_embeddings = self.sess.run(self.target_item_embeddings)
        return target_embeddings



    def trainProcess(self,next_batch_element):

            # 1：对下一批数据 next_batch_element，放入autoencoder训练

            _, encoded_, decoded_, autoEncoderLoss_ = self.sess.run([self.train, self.encoded, self.decoded, self.loss], feed_dict=next_batch_element)

            # print('autoEncoder train loss: %.4f\n' % autoEncoderLoss_, flush=True)

            #2: 得到autoencoder的输出，和标签（正样例的标签是1，负样例的标签是0）进行比较，更新encoder的参数

            '''Compute behavior norms'''
            self.batch_behaviors_matrix_inner_product = tf.matmul(self.encoded,
                                                                  self.encoded,
                                                                  transpose_a=False,
                                                                  transpose_b=True)
            self.reduce_sum_inner_product_matrix = tf.reduce_sum(self.batch_behaviors_matrix_inner_product, axis=(2, 3))
            self.batch_behaviors_norms = tf.sqrt(self.reduce_sum_inner_product_matrix)

            '''
            Objective function 1:
            When both positive and negative behaviors are observed.
            There is a numerical label for each behavior.
            '''
            self.batch_behaviors_success_rates = tf.tanh(self.batch_behaviors_norms / 2)
            self.l1_loss = -tf.reduce_sum(self.batch_behaviors_labels * tf.log(self.batch_behaviors_success_rates))

            '''
            Objective function 2:
            When only positive behaviors are observed,
            Labels for positive behaviors are 1, and negative behaviors have label 0.
            '''
            # self.batch_behaviors_success_rates = tf.tanh(self.batch_behaviors_norms / 2)
            # self.observed_behaviors_loss = self.batch_behaviors_labels * tf.log(self.batch_behaviors_success_rates)

            # self.negative_sampled_behaviors_loss = \
                # (1 - self.batch_behaviors_labels) * tf.log(tf.tanh(tf.reciprocal(self.batch_behaviors_norms) / 2))

            # self.l1_loss = -tf.reduce_sum(self.observed_behaviors_loss + self.negative_sampled_behaviors_loss)


            '''Optimizer and training operation'''
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.l1_loss)

            _, l1_loss_ , l1_lables, l1_encoded_, l1_product, l1_rates, l1_norms= self.sess.run([self.train_op, self.l1_loss,self.batch_behaviors_labels,self.encoded,self.batch_behaviors_matrix_inner_product,self.batch_behaviors_success_rates,self.batch_behaviors_norms], feed_dict=next_batch_element)

            # print("--l1_encoded_-shape--"*3)
            # print(l1_encoded_.shape) #(1,6,27,2)
            # print("--l1_encoded_---"*3)
            # print(l1_encoded_)

            # print("--l1_lables-shape--"*3)
            # print(l1_lables.shape) #(1,6)

            # print("--l1_product-shape--"*3)
            # print(l1_product.shape) #(1,6,27,27)

            # print("--l1_product---"*3)
            # print(l1_product)

            # print("--l1_norms-shape--"*3)
            # print(l1_norms.shape) #(1,6)

            # print("--l1_norms---"*3)
            # print(l1_norms)

            # exit()

            # print("--norm---"*3)
            # print(norms)
            # print("--l1_rates---"*3)
            # print(l1_rates)
            print(' L1 train loss: {:.4f};\tautoEncoder train loss: {:.4f}\n'.format(l1_loss_,autoEncoderLoss_), flush=True)

    #========================先前的代码==============================

            # visualize in 3D plot
            # view_data = self.test_x[:200]
            # encoded_data = self.sess.run(self.encoded, {tf_x: view_data})
            # fig = plt.figure(2); ax = Axes3D(fig)
            # X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
            # for x, y, z, s in zip(X, Y, Z, test_y):
            #     c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
            # ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
            # plt.show()


if __name__ == "__main__":
    args={}
    ae=AutoEncoderModel(args)
    ae.trainProcess()