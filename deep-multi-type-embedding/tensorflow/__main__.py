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

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--itemlist', required=False, help='the input file of items list')

    parser.add_argument('--behaviorlist', required=False, help='the input file of behaviors list')
    parser.add_argument('--evaluatebehaviorlist', required=False, help='the evaulate input file of items list')

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
    parser.add_argument('--evaluateNegative', default=0, type=int,
                        help='the number of negative samples used in negative sampling on evaluate stage; the default is 0')
    parser.add_argument('--samples', default=1, type=int,
                        help='the total number of training samples (*Thousand); the default is 1')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='the mini-batch size of the stochastic gradient descent; the default is 1')
    parser.add_argument('--aeLR', default=0.0004, type=float,
                        help='the starting value of the learning rate; the default is 0.025')
    parser.add_argument('--l1LR', default=0.0009, type=float,
                        help='the starting value of the learning rate; the default is 0.025')
    parser.add_argument('--threads', default=10, type=int,
                        help='the total number of threads used')
    parser.add_argument('--runMode', default='train')
    parser.add_argument('--testCount', default='1',type=int,help='the number of iter')
    args = parser.parse_args()

    if args.runMode == 'train':
        train(args)
    elif args.runMode == 'test':
        evaluate(args)



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

    #restore weitht

    ae.reStore(logdir='checkpoint/')


    #准备训练数据
    sampling_cum_time = 0
    training_cum_time = 0

    samples_number = args.samples * 1000

    # Report progress every 0.1% checkpoint
    checkpointNumber = 1
    # checkpointNumber = samples_number // 1000

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

        # print("输入的item类型：")
        # print(np.array(batch_behaviors_item_indices).shape) #(1,6,27)

        # print("输入的lable类型：")
        # print(np.array(batch_behaviors_labels).shape) #(1,6)

        # exit()

        '''Format mini-batch input to line model and execute one train operation'''
        feed_dict = {ae.batch_behaviors_item_indices: batch_behaviors_item_indices,
                     ae.batch_behaviors_item_type_indices: batch_behaviors_item_type_indices,
                     ae.batch_behaviors_labels: batch_behaviors_labels,
                     ae.learning_rate: args.l1LR}


        t_s = time.time()

        ae.trainProcess(feed_dict,sample_index) #对某一批数据进行训练

        t_e = time.time()
        training_cum_time += (t_e - t_s) #训练的时间

        '''Report progress'''
        if not sample_index % checkpointNumber:
            avg_loop_time = (sampling_cum_time + training_cum_time) / (sample_index + 1)
            etc = (samples_number - sample_index) * avg_loop_time
            progress = sample_index / samples_number
            print(' Progress: {:7.1%};\tETC: {:5.1f} min\n'.
                  format(progress, (etc / 60)), end='\r\n', flush=True)

        print('\n %s steps Training complete!'%(sample_index+1))

        if not (sample_index) % 999: #每1000次，evaulate一次
            evaluateInTrainProcess(args,ae,sample_index)




    '''Write out target embeddings'''
    print('---' * 30)
    print('Writing out embeddings...', end='\t', flush=True)
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


def EasyTanh(x):
    # if x > 3.0: return 1.0
    return np.tanh(x)

def evaluateInTrainProcess(args,ae,iter):

    print("evaluate start ======")

    print('================args=================')
    print(args)
    print('=====================================')


        #准备训练数据
    sampling_cum_time = 0

    '''Create and initialize dataset manager'''
    t_s = time.time()
    print('In evaluate stage-- Create new behavior dataset manager:')
    dataset_manager = BehaviorDatasetManager()
    dataset_manager.read_itemlist_file(itemlist_file=args.itemlist)
    dataset_manager.read_behaviorlist_file(behaviorlist_file=args.evaluatebehaviorlist)
    print('---' * 30)




    args.number_of_items = dataset_manager.number_of_items #item的数量，在构建graph的时候会用这个参数

    t_e = time.time()
    initialization_time = (t_e - t_s) #数据的初始化时间，不包含模型的初始化时间


    evaluate_samples_number = dataset_manager.number_of_behaviors # behaviros 的数量

    # Report progress every 0.1% checkpoint
    checkpoint = 10

    print("evaluate_samples_number:"+str(evaluate_samples_number))

    r_b_pos=[]
    lable_pos=[]

    for sample_index in range(evaluate_samples_number):
        '''Sample mini-batch of behaviors'''
        t_s = time.time()
        sampled_results = dataset_manager.get_nextBatch_evaluateBehaviors(sample_index,batch_size=args.batch_size,
                                                                 negative=args.evaluateNegative,
                                                                 mode=args.mode)
        batch_behaviors_item_indices = sampled_results[0]
        batch_behaviors_item_type_indices = sampled_results[1]
        batch_behaviors_labels = sampled_results[2]
        t_e = time.time()
        sampling_cum_time += (t_e - t_s)

        # print("输入的item类型：")
        # print(np.array(batch_behaviors_item_indices).shape) #(1,1,32)

        # print("输入的lable类型：")
        # print(np.array(batch_behaviors_labels).shape) #(1,1)

        lable_pos.append(np.array(batch_behaviors_labels))

        '''Format mini-batch input to line model and execute one train operation'''
        feed_dict = {ae.batch_behaviors_item_indices: batch_behaviors_item_indices,
                     ae.batch_behaviors_item_type_indices: batch_behaviors_item_type_indices,
                     ae.batch_behaviors_labels: batch_behaviors_labels,
                     ae.learning_rate: args.l1LR}


        encoded_=ae.evaluateProcess(feed_dict) #对某一批数据进行测试

        encoded_=np.array(encoded_)

        # print("--输出的encoded_-shape--"*3)
        # print(np.array(encoded_).shape) #(1,1,30)

        # print("--encoded_---"*3)
        # print(encoded_)

        # exit()

        norm_b = np.linalg.norm(encoded_,2,axis=2)
        # norm_b_pos.append(norm_b)
        r_b = EasyTanh(norm_b/2)
        r_b_pos.append(r_b)

        # print(r_b.shape) #(1,1) 后面的1 为negative的数量决定
        # print("r_b:"+str(r_b))
        # exit()

        '''Report progress'''
        if not sample_index % checkpoint:
            progress = sample_index / evaluate_samples_number
            print('Progress: {:7.1%}'.format(progress), end='\r\n', flush=True)

    print('\nEvaluate complete!')


    print("SUMMARY of  evaluate:")

    MAE=0
    RMSE=0

    for index,value in enumerate(r_b_pos):

        lable=lable_pos[index]
        sub= np.fabs(value-lable)
        MAE+=np.sum(sub)

        temp=np.square(sub)
        RMSE+=np.sum(temp)
        # print(sub.shape)
        # exit()

    MAE=MAE/len(r_b_pos)
    RMSE=np.sqrt(RMSE/len(r_b_pos))

    MAEStr="MAE:"+str(MAE)
    RMSEStr="RMSE:"+str(RMSE)
    print(MAEStr)
    print(RMSEStr)


    print("Writing to file evaluate:")


    with open("evaluate.txt","a+") as fp:
        fp.write("\n")

        fp.write(str(iter))
        fp.write("\t\t")
        fp.write(str(MAE))
        fp.write("\t\t")
        fp.write(str(RMSE))

def addLineToFile():
    with open("evaluate.txt","a+") as fp:
        fp.write("\n")

        fp.write("====================test in outer python==negative sample： 1======================")

'''这个函数配合run.py中的代码，用命令行执行的时候会调用'''
def evaluate(args):

    addLineToFile()

    print("evaluate start ======")

    print('================args=================')
    print(args)
    print('=====================================')


        #准备训练数据
    sampling_cum_time = 0

    '''Create and initialize dataset manager'''
    t_s = time.time()
    print('In evaluate stage-- Create new behavior dataset manager:')
    dataset_manager = BehaviorDatasetManager()
    dataset_manager.read_itemlist_file(itemlist_file=args.itemlist)
    dataset_manager.read_behaviorlist_file(behaviorlist_file=args.evaluatebehaviorlist)
    print('---' * 30)




    args.number_of_items = dataset_manager.number_of_items #item的数量，在构建graph的时候会用这个参数

    t_e = time.time()
    initialization_time = (t_e - t_s) #数据的初始化时间，不包含模型的初始化时间


    '''针对所有保存的训练过程中的参数列表，进行测试，看哪一个是最好的'''
    print("reStore start")
    ckpt = tf.train.get_checkpoint_state("checkpoint/")
    print("=================ckpt:")
    print(ckpt)


    print('Create AutoEncoderModel model computation graph in Tensorflow:')
    ae=AutoEncoderModel(args)

    #加载模型，针对每一个保存的模型，运算准确率等信息
    #这一部分是有多个模型文件时，对所有模型进行测试验证
    for path in ckpt.all_model_checkpoint_paths:

        global_step=path.split('/')[-1].split('-')[-1]




        #restore weitht
        ae.reStore(path)


        evaluate_samples_number = dataset_manager.number_of_behaviors # behaviros 的数量

        # Report progress every 0.1% checkpoint
        checkpoint = 10

        print("evaluate_samples_number:"+str(evaluate_samples_number))

        r_b_pos=[]
        lable_pos=[]

        for sample_index in range(evaluate_samples_number):
            '''Sample mini-batch of behaviors'''
            t_s = time.time()
            sampled_results = dataset_manager.get_nextBatch_evaluateBehaviors(sample_index,batch_size=args.batch_size,
                                                                     negative=args.negative,
                                                                     mode=args.mode)
            batch_behaviors_item_indices = sampled_results[0]
            batch_behaviors_item_type_indices = sampled_results[1]
            batch_behaviors_labels = sampled_results[2]
            t_e = time.time()
            sampling_cum_time += (t_e - t_s)

            # print("输入的item类型：")
            # print(np.array(batch_behaviors_item_indices).shape) #(1,1,32)

            # print("输入的lable类型：")
            # print(np.array(batch_behaviors_labels).shape) #(1,1)

            lable_pos.append(np.array(batch_behaviors_labels))

            '''Format mini-batch input to line model and execute one train operation'''
            feed_dict = {ae.batch_behaviors_item_indices: batch_behaviors_item_indices,
                         ae.batch_behaviors_item_type_indices: batch_behaviors_item_type_indices,
                         ae.batch_behaviors_labels: batch_behaviors_labels,
                         ae.learning_rate: args.l1LR}


            encoded_=ae.evaluateProcess(feed_dict) #对某一批数据进行测试

            encoded_=np.array(encoded_)

            # print("--输出的encoded_-shape--"*3)
            # print(np.array(encoded_).shape) #(1,1,30)

            # print("--encoded_---"*3)
            # print(encoded_)

            # exit()

            norm_b = np.linalg.norm(encoded_,2,axis=2)
            # norm_b_pos.append(norm_b)
            r_b = EasyTanh(norm_b/2)
            r_b_pos.append(r_b)

            # print(r_b.shape) #(1,1) 后面的1 为negative的数量决定
            # print("r_b:"+str(r_b))
            # exit()

            '''Report progress'''
            if not sample_index % checkpoint:
                progress = sample_index / evaluate_samples_number
                print(' Progress: {:7.1%}'.format(progress), end='\r\n', flush=True)

        print('\nEvaluate complete!')


        print("SUMMARY of  evaluate:")

        MAE=0
        RMSE=0

        for index,value in enumerate(r_b_pos):

            lable=lable_pos[index]
            sub= np.fabs(value-lable)
            MAE+=np.sum(sub)

            temp=np.square(sub)
            RMSE+=np.sum(temp)
            # print(sub.shape)
            # exit()

        MAE=MAE/len(r_b_pos)
        RMSE=np.sqrt(RMSE/len(r_b_pos))

        MAEStr="MAE:"+str(MAE)
        RMSEStr="RMSE:"+str(RMSE)
        print(MAEStr)
        print(RMSEStr)


        print("Writing to file evaluate:")


        with open("evaluate.txt","a+") as fp:
            fp.write("\n")

            fp.write(str(global_step))
            fp.write("\t\t")
            fp.write(str(MAE))
            fp.write("\t\t")
            fp.write(str(RMSE))






#=======================在外面执行时候的代码，代码没有调试，有bug================================


def sample_Z(m, n): #生成维度为[m, n]的随机噪声作为生成器G的输入
    return np.random.uniform(-1., 1., size=[m, n])

def save(saver, sess, logdir, step): #保存模型的save函数
   model_name = 'model' #模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #保存路径
   saver.save(sess, checkpoint_path, global_step=step) #保存模型
   print('The checkpoint has been created.')


if __name__ == "__main__":
  sys.exit(main())

  ## Usage train
'''
python tensorflow --encoderSize 30 --itemlist tensorflow/data/itemlist.txt --behaviorlist tensorflow/data/behaviorlist.txt --output tensorflow/embeddings.mode1 --mode 1 --threads 8 --samples 1

'''

## Usage test
'''
python tensorflow --encoderSize 30 --negative 0 --itemlist tensorflow/data/itemlist.txt --behaviorlist tensorflow/data/evaluate.txt --output tensorflow/embeddings.mode1 --mode 1 --threads 8  --runMode test

'''