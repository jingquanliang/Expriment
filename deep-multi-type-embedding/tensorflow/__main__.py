#! /usr/bin/env python
# -*- coding: utf-8 -*-



import tensorflow as tf #导入tensorflow
from tensorflow.examples.tutorials.mnist import input_data #导入手写数字数据集
import numpy as np #导入numpy
import matplotlib.pyplot as plt #plt是绘图工具，在训练过程中用于输出可视化结果
import matplotlib.gridspec as gridspec #gridspec是图片排列工具，在训练过程中用于输出可视化结果
import os #导入os
import sys


from dataset_behavior import BehaviorDatasetManager

import time

import argparse

from GANModel import GANModel
from AutoEncoderModel import AutoEncoderModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--itemlist', required=False, help='the input file of items list')
    parser.add_argument('--behaviorlist', required=False, help='the input file of behaviors list')
    parser.add_argument('--output', required=False, help='the output file of context item embeddings')
    parser.add_argument('--size', default=128, type=int,
                        help='the dimension of the embedding; the default is 128')
    parser.add_argument('--encoderSize', default=30, type=int,
                        help='the dimension of the encoder size; the default is 30')
    parser.add_argument('--mode', default='1', choices=['1', '2'],
                        help='the negative sampling method used.'
                             '1 for size-constrained, 2 for type-constrained; the default is 1')
    parser.add_argument('--negative', default=5, type=int,
                        help='the number of negative samples used in negative sampling; the default is 5')
    parser.add_argument('--samples', default=1, type=int,
                        help='the total number of training samples (*Thousand); the default is 1')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='the mini-batch size of the stochastic gradient descent; the default is 1')
    parser.add_argument('--rho', default=0.025, type=float,
                        help='the starting value of the learning rate; the default is 0.025')
    parser.add_argument('--threads', default=10, type=int,
                        help='the total number of threads used')
    parser.add_argument('--runMode', default='train')
    args = parser.parse_args()

    if args.runMode == 'train':
        train(args)
    elif args.runMode == 'test':
        test(args)



def train(args):


    print('================args=================')
    print(args)
    print('=====================================')


    '''Create and initialize dataset manager'''
    t_s = time.time()
    print('Create new behavior dataset manager:')
    dataset_manager = BehaviorDatasetManager()
    dataset_manager.read_itemlist_file(itemlist_file=args.itemlist)
    dataset_manager.read_behaviorlist_file(behaviorlist_file=args.behaviorlist)
    print('---' * 30)


    '''数据初始化'''

    args.number_of_items = dataset_manager.number_of_items #item的数量，在构建graph的时候会用这个参数

    t_e = time.time()
    initialization_time = (t_e - t_s) #数据的初始化时间，不包含模型的初始化时间


    '''通过gan得到负样例数据'''
    # gan=GANModel(args)
    # gan.train()

    print('Create AutoEncoderModel model computation graph in Tensorflow:')
    ae=AutoEncoderModel(args)


    curr_learning_rate=args.rho
    #准备训练数据
    sampling_cum_time = 0
    training_cum_time = 0

    samples_number = args.samples * 1000

    # Report progress every 0.1% checkpoint
    checkpoint = samples_number // 1000

    for sample_index in range(samples_number):
        '''Sample mini-batch of behaviors'''
        t_s = time.time()
        sampled_results = dataset_manager.sample_batch_behaviors(batch_size=args.batch_size,
                                                                 negative=args.negative,
                                                                 mode=args.mode)
        batch_behaviors_item_indices = sampled_results[0]
        batch_behaviors_item_type_indices = sampled_results[1]
        batch_behaviors_labels = sampled_results[2]
        t_e = time.time()
        sampling_cum_time += (t_e - t_s)

        '''Format mini-batch input to line model and execute one train operation'''
        feed_dict = {ae.batch_behaviors_item_indices: batch_behaviors_item_indices,
                     ae.batch_behaviors_item_type_indices: batch_behaviors_item_type_indices,
                     ae.batch_behaviors_labels: batch_behaviors_labels,
                     ae.learning_rate: args.rho}


        t_s = time.time()

        ae.trainProcess(feed_dict) #对某一批数据进行训练

        t_e = time.time()
        training_cum_time += (t_e - t_s) #训练的时间

        '''Report progress'''
        if not sample_index % checkpoint:
            avg_loop_time = (sampling_cum_time + training_cum_time) / (sample_index + 1)
            etc = (samples_number - sample_index) * avg_loop_time
            progress = sample_index / samples_number
            print(' Current rho: {:9.5f};\tProgress: {:7.1%};\tETC: {:5.1f} min\n'.
                  format(curr_learning_rate, progress, (etc / 60)), end='\r', flush=True)

        '''Update learning rate'''
        if curr_learning_rate < args.rho * 0.0001:
            # Set minial learning rate
            curr_learning_rate = args.rho * 0.0001
        else:
            curr_learning_rate = args.rho * (1 - (sample_index / samples_number))

    print('\nTraining complete!')


    '''Write out target embeddings'''
    print('---' * 30)
    print('Writing out...', end='\t', flush=True)
    t_s = time.time()
    target_embeddings = ae.getTarget_embeddings()
    dataset_manager.output_embedding(target_embeddings, args.output, mode='txt')
    t_e = time.time()
    output_time = t_e - t_s
    print('Done! ({:.2f} sec)'.format(output_time))
    print('---' * 30)

    '''Print runtime summary'''
    total_time = initialization_time + sampling_cum_time + training_cum_time + output_time
    print('SUMMARY:')
    print('Total elapsed time: {:.1f} min'.format(total_time / 60))
    print('\tInitialization: {:.1f} min ({:.2%})'.
          format(initialization_time / 60, (initialization_time / total_time)))
    print('\tSampling: {:.1f} min ({:.2%})'.format(sampling_cum_time / 60, (sampling_cum_time / total_time)))
    print('\tTraining: {:.1f} min ({:.2%})'.format(training_cum_time / 60, (training_cum_time / total_time)))
    print('\tOutput: {:.1f} min ({:.2%})'.format(output_time / 60, (output_time / total_time)))
    print('---' * 30)


## Usage
'''
python tensorflow --itemlist tensorflow/data/itemlist.txt --behaviorlist tensorflow/data/behaviorlist.txt --output tensorflow/embeddings.mode1 --mode 1 --threads 8

'''



#=======================在外面执行时候的代码，代码没有调试，有bug================================


def sample_Z(m, n): #生成维度为[m, n]的随机噪声作为生成器G的输入
    return np.random.uniform(-1., 1., size=[m, n])

def save(saver, sess, logdir, step): #保存模型的save函数
   model_name = 'model' #模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #保存路径
   saver.save(sess, checkpoint_path, global_step=step) #保存模型
   print('The checkpoint has been created.')

    # with tf.Session() as sess:

    #     mb_size = 128 #训练的batch_size

    #     Z_dim = 100 #生成器输入的随机噪声的列的维度

    #     mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True) #mnist是手写数字数据集

    #     summary_writer = tf.summary.FileWriter('snapshots/', graph=tf.get_default_graph()) #日志记录器


    #     sess = tf.Session() #会话层
    #     sess.run(tf.global_variables_initializer()) #初始化所有可训练参数

    #     if not os.path.exists('out/'): #初始化训练过程中的可视化结果的输出文件夹
    #         os.makedirs('out/')

    #     if not os.path.exists('snapshots/'): #初始化训练过程中的模型保存文件夹
    #         os.makedirs('snapshots/')

    #     saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50) #模型的保存器

    #     i = 0 #训练过程中保存的可视化结果的索引

    #     for it in range(1000000): #训练100万次
    #         if it % 1000 == 0: #每训练1000次就保存一下结果
    #             samples = sess.run(gan.G_sample, feed_dict={gan.Z: sample_Z(16, Z_dim)})

    #             fig = plot(samples) #通过plot函数生成可视化结果
    #             plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight') #保存可视化结果
    #             i += 1
    #             plt.close(fig)

    #         X_mb, _ = mnist.train.next_batch(mb_size) #得到训练一个batch所需的真实手写数字(作为判别器的输入)

    #         #下面是得到训练一次的结果，通过sess来run出来
    #         _, D_loss_curr, dreal_loss_sum_value, dfake_loss_sum_value, d_loss_sum_value = sess.run([gan.D_solver, gan.D_loss, gan.dreal_loss_sum, gan.dfake_loss_sum, gan.d_loss_sum], feed_dict={gan.X: X_mb, gan.Z: sample_Z(mb_size, Z_dim)})

    #         _, G_loss_curr, g_loss_sum_value = sess.run([gan.G_solver, gan.G_loss, gan.g_loss_sum], feed_dict={gan.Z: sample_Z(mb_size, Z_dim)})

    #         if it%100 ==0: #每过100次记录一下日志，可以通过tensorboard查看

    #             #merge_summary = tf.summary.merge_all()
    #             #summary_writer.add_summary(merge_summary, it)
    #             summary_writer.add_summary(dreal_loss_sum_value, it)  #输出的名字由上边的dreal_loss_sum_value定义中确定
    #             summary_writer.add_summary(dfake_loss_sum_value, it)
    #             summary_writer.add_summary(d_loss_sum_value, it)
    #             summary_writer.add_summary(g_loss_sum_value, it)

    #         if it % 1000 == 0: #每训练1000次输出一下结果
    #             save(self.saver, self.sess, 'snapshots/', it)
    #             print('Iter: {}'.format(it))
    #             print('D loss: {:.4}'. format(D_loss_curr))
    #             print('G_loss: {:.4}'.format(G_loss_curr))
    #             print()


if __name__ == "__main__":
  sys.exit(main())
