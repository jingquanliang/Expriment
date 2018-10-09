import tensorflow as tf #导入tensorflow
from tensorflow.examples.tutorials.mnist import input_data #导入手写数字数据集
import numpy as np #导入numpy
import matplotlib.pyplot as plt #plt是绘图工具，在训练过程中用于输出可视化结果
import matplotlib.gridspec as gridspec #gridspec是图片排列工具，在训练过程中用于输出可视化结果
import os #导入os

'''
模型损失函数的整体含义：让正样本（相邻节点）的目标函数值够大，同时让负样本（不相邻节点）的目标函数值小
'''
class GANModel:



    def __init__(self, args):

        # ss=self.xavier_init([100, 128])
        # print(ss)

        with tf.variable_scope('Generator'):

            self.Z = tf.placeholder(tf.float32, shape=[None, 100]) #Z表示生成器的输入(在这里是噪声)，是一个N行100列的矩阵

            self.G_W1 = tf.Variable(self.xavier_init([100, 128])) #表示使用xavier方式初始化的生成器的G_W1参数，是一个100行128列的矩阵
            self.G_b1 = tf.Variable(tf.zeros(shape=[128])) #表示全零方式初始化的生成器的G_b1参数，是一个长度为128的向量

            self.G_W2 = tf.Variable(self.xavier_init([128, 784])) #表示使用xavier方式初始化的生成器的G_W2参数，是一个128行784列的矩阵
            self.G_b2 = tf.Variable(tf.zeros(shape=[784])) #表示全零方式初始化的生成器的G_b2参数，是一个长度为784的向量

            self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2] #theta_G表示生成器的可训练参数集合



        with tf.variable_scope('Discriminator'):

            self.X = tf.placeholder(tf.float32, shape=[None, 784]) #X表示真的样本(即真实的手写数字)

            self.D_W1 = tf.Variable(self.xavier_init([784, 128])) #表示使用xavier方式初始化的判别器的D_W1参数，是一个784行128列的矩阵
            self.D_b1 = tf.Variable(tf.zeros(shape=[128])) #表示全零方式初始化的判别器的D_1参数，是一个长度为128的向量

            self.D_W2 = tf.Variable(self.xavier_init([128, 1])) #表示使用xavier方式初始化的判别器的D_W2参数，是一个128行1列的矩阵
            self.D_b2 = tf.Variable(tf.zeros(shape=[1])) ##表示全零方式初始化的判别器的D_1参数，是一个长度为1的向量

            self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2] #theta_D表示判别器的可训练参数集合



        self.G_sample = self.generator(self.Z) #取得生成器的生成结果
        self.D_real, self.D_logit_real = self.discriminator(self.X) #取得判别器判别的真实手写数字的结果
        self.D_fake, self.D_logit_fake = self.discriminator(self.G_sample) #取得判别器判别的生成的手写数字的结果

        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real))) #对判别器对真实样本的判别结果计算误差(将结果与1比较)

        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake))) #对判别器对虚假样本(即生成器生成的手写数字)的判别结果计算误差(将结果与0比较)

        self.D_loss = self.D_loss_real + self.D_loss_fake #判别器的误差

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.ones_like(self.D_logit_fake))) #生成器的误差(将判别器返回的对虚假样本的判别结果与1比较)


        self.dreal_loss_sum = tf.summary.scalar("dreal_loss", self.D_loss_real) #记录判别器判别真实样本的误差
        self.dfake_loss_sum = tf.summary.scalar("dfake_loss", self.D_loss_fake) #记录判别器判别虚假样本的误差
        self.d_loss_sum = tf.summary.scalar("d_loss", self.D_loss) #记录判别器的误差
        self.g_loss_sum = tf.summary.scalar("g_loss", self.G_loss) #记录生成器的误差


        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D) #判别器的训练器

        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G) #生成器的训练器


        self.summary_writer = tf.summary.FileWriter('snapshots/', graph=tf.get_default_graph()) #日志记录器



    def save(self,saver, sess, logdir, step): #保存模型的save函数
       model_name = 'model' #模型名前缀
       checkpoint_path = os.path.join(logdir, model_name) #保存路径
       saver.save(sess, checkpoint_path, global_step=step) #保存模型
       print('The checkpoint has been created.')

    def xavier_init(self,size): #初始化参数时使用的xavier_init函数
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.) #初始化标准差
        return tf.random_normal(shape=size, stddev=xavier_stddev) #返回初始化的结果

    def sample_Z(self,m, n): #生成维度为[m, n]的随机噪声作为生成器G的输入
        return np.random.uniform(-1., 1., size=[m, n])


    def generator(self,z): #生成器，z的维度为[N, 100]
        self.G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1) #输入的随机噪声乘以G_W1矩阵加上偏置G_b1，G_h1维度为[N, 128]
        self.G_log_prob = tf.matmul(self.G_h1, self.G_W2) + self.G_b2 #G_h1乘以G_W2矩阵加上偏置G_b2，G_log_prob维度为[N, 784]
        self.G_prob = tf.nn.sigmoid(self.G_log_prob) #G_log_prob经过一个sigmoid函数，G_prob维度为[N, 784]

        return self.G_prob #返回G_prob


    def discriminator(self,x): #判别器，x的维度为[N, 784]
        self.D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1) #输入乘以D_W1矩阵加上偏置D_b1，D_h1维度为[N, 128]
        self.D_logit = tf.matmul(self.D_h1, self.D_W2) + self.D_b2 #D_h1乘以D_W2矩阵加上偏置D_b2，D_logit维度为[N, 1]
        self.D_prob = tf.nn.sigmoid(self.D_logit) #D_logit经过一个sigmoid函数，D_prob维度为[N, 1]

        return self.D_prob, self.D_logit #返回D_prob, D_logit



    def plot(self,samples): #保存图片时使用的plot函数
        fig = plt.figure(figsize=(4, 4)) #初始化一个figsize大小的图片
        gs = gridspec.GridSpec(4, 4) #调整子图的位置
        gs.update(wspace=0.05, hspace=0.05) #置子图间的间距

        for i, sample in enumerate(samples): #依次将16张子图填充进需要保存的图像
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

    def train(self): #进行训练  #X表示真的样本(即真实的手写数字),Z表示随机噪声作为生成器G的输入

        mb_size = 128 #训练的batch_size

        Z_dim = 100 #生成器输入的随机噪声的列的维度

        mnist = input_data.read_data_sets('mnist', one_hot=True) #mnist是手写数字数据集

        self.sess = tf.Session() #会话层
        self.sess.run(tf.global_variables_initializer()) #初始化所有可训练参数

        if not os.path.exists('out/'): #初始化训练过程中的可视化结果的输出文件夹
            os.makedirs('out/')

        if not os.path.exists('snapshots/'): #初始化训练过程中的模型保存文件夹
            os.makedirs('snapshots/')

        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50) #模型的保存器

        i = 0 #训练过程中保存的可视化结果的索引

        for it in range(1000000): #训练100万次
            if it % 1000 == 0: #每训练1000次就保存一下结果
                samples = self.sess.run(self.G_sample, feed_dict={self.Z: self.sample_Z(16, Z_dim)})

                fig = self.plot(samples) #通过plot函数生成可视化结果
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight') #保存可视化结果
                i += 1
                plt.close(fig)

            X_mb, _ = mnist.train.next_batch(mb_size) #得到训练一个batch所需的真实手写数字(作为判别器的输入)

            #下面是得到训练一次的结果，通过sess来run出来
            _, D_loss_curr, dreal_loss_sum_value, dfake_loss_sum_value, d_loss_sum_value = self.sess.run([self.D_solver, self.D_loss, self.dreal_loss_sum, self.dfake_loss_sum, self.d_loss_sum], feed_dict={self.X: X_mb, self.Z: self.sample_Z(mb_size, Z_dim)})

            _, G_loss_curr, g_loss_sum_value = self.sess.run([self.G_solver, self.G_loss, self.g_loss_sum], feed_dict={self.Z: self.sample_Z(mb_size, Z_dim)})

            if it%100 ==0: #每过100次记录一下日志，可以通过tensorboard查看

                #merge_summary = tf.summary.merge_all()
                #summary_writer.add_summary(merge_summary, it)
                self.summary_writer.add_summary(dreal_loss_sum_value, it)  #输出的名字由上边的dreal_loss_sum_value定义中确定
                self.summary_writer.add_summary(dfake_loss_sum_value, it)
                self.summary_writer.add_summary(d_loss_sum_value, it)
                self.summary_writer.add_summary(g_loss_sum_value, it)

            if it % 1000 == 0: #每训练1000次输出一下结果
                self.save(self.saver, self.sess, 'snapshots/', it)
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()


