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
import os #导入os

tf.set_random_seed(1)


class AutoEncoderModel:

    def __init__(self, args):


        # Hyper Parameters
        self.BATCH_SIZE = args.batch_size
        self.LR = args.aeLR         # learning rate of endcoder-decoder
        self.l1LR= args.l1LR
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
        self.en0 = tf.layers.dense(self.batch_behaviors_weighted_representations, 70, tf.nn.relu)
        self.en1 = tf.layers.dense(self.en0, 50, tf.nn.relu)
        self.en2 = tf.layers.dense(self.en1, 40, tf.nn.relu)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.00001)
        # self.encoded = tf.layers.dense(self.en2, args.encoderSize,tf.nn.relu,kernel_regularizer=regularizer)
        self.encoded = tf.layers.dense(self.en2, args.encoderSize,tf.nn.tanh)

        # decoder
        self.de0 = tf.layers.dense(self.encoded, 40, tf.nn.relu)
        self.de1 = tf.layers.dense(self.de0, 50, tf.nn.relu)
        self.de2 = tf.layers.dense(self.de1, 70, tf.nn.relu)
        self.decoded = tf.layers.dense(self.de2, args.size, tf.nn.relu)

        self.loss = tf.losses.mean_squared_error(labels=self.batch_behaviors_weighted_representations, predictions=self.decoded)
        self.train = tf.train.AdamOptimizer(self.LR).minimize(self.loss)


        self.trainInitalization(args)

        # self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50) #模型的保存器

        self.saver = tf.train.Saver(max_to_keep=100000)

        self.dashBoard()

        self.sessInitalization(args)#初始化sess

    def dashBoard(self):

        # summary_writer.add_summary(dreal_loss_sum_value, it)  #输出的名字由上边的dreal_loss_sum_value定义中确定
        self.ae_loss_summ = tf.summary.scalar("ae_loss", self.loss) #记录判别器判别真实样本的误差
        self.l1_loss_summ = tf.summary.scalar("l1_loss", self.l1_loss) #记录判别器判别虚假样本的误差

        self.summary_writer = tf.summary.FileWriter('snapshots/', graph=tf.get_default_graph()) #日志记录器

    def trainInitalization(self,args):


            self.batch_behaviors_matrix_inner_product=tf.multiply(self.encoded,self.encoded)

            self.reduce_sum_inner_product_matrix = tf.reduce_sum(self.batch_behaviors_matrix_inner_product, axis=(2))
            self.batch_behaviors_norms = tf.sqrt(self.reduce_sum_inner_product_matrix)

            '''
            Objective function 1:
            When both positive and negative behaviors are observed.
            There is a numerical label for each behavior.
            '''
            # self.batch_behaviors_success_rates = tf.tanh(self.batch_behaviors_norms / 2)
            # self.l1_loss = -tf.reduce_sum(self.batch_behaviors_labels * tf.log(self.batch_behaviors_success_rates))

            '''
            Objective function 2:
            When only positive behaviors are observed,
            Labels for positive behaviors are 1, and negative behaviors have label 0.
            '''
            self.batch_behaviors_success_rates = tf.tanh(self.batch_behaviors_norms / 2)
            self.observed_behaviors_loss = self.batch_behaviors_labels * tf.log(self.batch_behaviors_success_rates)

            self.negative_sampled_behaviors_loss = \
                (1 - self.batch_behaviors_labels) * tf.log(tf.tanh(tf.reciprocal(self.batch_behaviors_norms) / 2))


            self.l1_loss = -tf.reduce_sum(self.observed_behaviors_loss + self.negative_sampled_behaviors_loss)


            # l2_loss = tf.losses.get_regularization_loss()

            # self.l1_loss +=l2_loss


            '''Optimizer and training operation'''
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optimizer = tf.train.AdamOptimizer(self.l1LR)
            self.train_op = self.optimizer.minimize(self.l1_loss)


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
                                                      initializer=tf.random_uniform_initializer(minval=-1,
                                                                                                maxval=1),
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
        self.batch_behaviors_weighted_representations1 = tf.einsum('ijtd,ijt->ijtd',
                                                                  self.batch_behaviors_item_embeddings,
                                                                  self.batch_behaviors_item_weights)

        self.batch_behaviors_weighted_representations=tf.reduce_sum(self.batch_behaviors_weighted_representations1,2)

    def getTarget_embeddings(self):

        '''Write out target embeddings'''
        target_embeddings = self.sess.run(self.target_item_embeddings)
        return target_embeddings

    def save(self, saver, sess, logdir, step): #保存模型的save函数
       model_name = 'deep-model' #模型名前缀
       checkpoint_path = os.path.join(logdir, model_name) #保存路径
       saver.save(sess, checkpoint_path, global_step=step) #保存模型
       print('The checkpoint has been created.')


    def reStore(self,path=None,logdir="checkpoint/"): #回复模型参数的函数,回复模型的最新的参数

        # destroy previous net
        # tf.reset_default_graph()

        print("reStore start")
        if path:
            self.saver.restore(self.sess, path)
        else:
            ckpt = tf.train.get_checkpoint_state(logdir)
            print("=================ckpt:")
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("reStore complete")
                print("===="*20)

    def trainProcess(self,next_batch_element,it):

            # 1：对下一批数据 next_batch_element，放入autoencoder训练

            _, encoded_, decoded_, autoEncoderLoss_,autoEncoderLoss_summ = self.sess.run([self.train, self.encoded, self.decoded, self.loss,self.ae_loss_summ], feed_dict=next_batch_element)


            #2: 得到autoencoder的输出，和标签（正样例的标签是1，负样例的标签是0）进行比较，更新encoder的参数

            _, l1_loss_, l1_loss_summ , l1_lables, l1_encoded_, l1_product, l1_rates, l1_norms= self.sess.run([self.train_op, self.l1_loss, self.l1_loss_summ,self.batch_behaviors_labels,self.encoded,self.batch_behaviors_matrix_inner_product,self.batch_behaviors_success_rates,self.batch_behaviors_norms], feed_dict=next_batch_element)



            self.summary_writer.add_summary(autoEncoderLoss_summ, it)  #输出的名字由上边的函数定义中确定
            self.summary_writer.add_summary(l1_loss_summ, it)

            print(' L1 train loss: {:.4f};\tautoEncoder train loss: {:.4f}'.format(l1_loss_,autoEncoderLoss_), end='\r\n', flush=True)


            # self.save(self.saver, self.sess, 'checkpoint/', it)

            if it % 1000 == 999: #每训练1000次输出一下结果
                print('---' * 30)
                print('save the weights in iter:'+str(it))
                self.save(self.saver, self.sess, 'checkpoint/', it)
                print('Iter: {}'.format(it))
                print()
                print('---' * 30)

            # print("--decoded_-shape--"*3)
            # print(decoded_.shape) #(1,6,27,2)
            # print("--decoded_---"*3)
            # print(decoded_)

            # exit()

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


    def evaluateProcess(self,next_batch_element):

            # 1：对下一批数据 next_batch_element，放入autoencoder，得到encode

            encoded_,wrepres,itembedding,itemWeight= self.sess.run([self.encoded,self.batch_behaviors_weighted_representations,self.batch_behaviors_item_embeddings,self.batch_behaviors_item_weights], feed_dict=next_batch_element)

            # encoded_= self.sess.run([self.encoded], feed_dict=next_batch_element)


            # print("--输出的encoded_-shape--"*3)
            # print(np.array(encoded_).shape)
            # print(wrepres.shape)
            # print(itembedding.shape)
            # print(itemWeight.shape)
            # exit()
            return encoded_



if __name__ == "__main__":
    args={}
    ae=AutoEncoderModel(args)
    ae.trainProcess()