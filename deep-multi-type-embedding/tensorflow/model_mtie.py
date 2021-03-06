"""
Multi-Type Itemset Embedding model
"""

import argparse
import random
import time

import numpy as np
import tensorflow as tf

from dataset_behavior import BehaviorDatasetManager

__author__ = "Daheng Wang"
__email__ = "dwang8@nd.edu"


class ModelMTIE:

    def __init__(self, args):
        """
        Computation graph of MTIE model
        """
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

        '''Compute behavior norms'''
        self.batch_behaviors_matrix_inner_product = tf.matmul(self.batch_behaviors_weighted_representations,
                                                              self.batch_behaviors_weighted_representations,
                                                              transpose_a=False,
                                                              transpose_b=True)
        self.reduce_sum_inner_product_matrix = tf.reduce_sum(self.batch_behaviors_matrix_inner_product, axis=(2, 3))
        self.batch_behaviors_norms = tf.sqrt(self.reduce_sum_inner_product_matrix)




        '''
        Objective function 1:
        When both positive and negative behaviors are observed.
        There is a numerical label for each behavior.
        '''
        # self.batch_behaviors_success_rates = tf.tanh(self.batch_behaviors_norms / 2)
        # self.loss = -tf.reduce_sum(self.batch_behaviors_labels * tf.log(self.batch_behaviors_success_rates))

        '''
        Objective function 2:
        When only positive behaviors are observed,
        Labels for positive behaviors are 1, and negative behaviors have label 0.
        '''
        self.batch_behaviors_success_rates = tf.tanh(self.batch_behaviors_norms / 2)
        self.observed_behaviors_loss = self.batch_behaviors_labels * tf.log(self.batch_behaviors_success_rates)
        self.negative_sampled_behaviors_loss = \
            (1 - self.batch_behaviors_labels) * tf.log(tf.tanh(tf.reciprocal(self.batch_behaviors_norms) / 2))
        self.loss = -tf.reduce_sum(self.observed_behaviors_loss + self.negative_sampled_behaviors_loss)

        '''Placeholder for Learning rate'''
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)

        '''Optimizer and training operation'''
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        self.l1_loss_summ = tf.summary.scalar("modle-mite-loss", self.loss) #记录判别器判别真实样本的误差
        self.merge_op = tf.summary.merge_all()                       # operation to merge all summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--itemlist', required=True, help='the input file of items list')
    parser.add_argument('--behaviorlist', required=True, help='the input file of behaviors list')
    parser.add_argument('--output', required=True, help='the output file of context item embeddings')
    parser.add_argument('--size', default=128, type=int,
                        help='the dimension of the embedding; the default is 128')
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
    args = parser.parse_args()

    return args


def train_mtie(args):
    print('---' * 30)
    print('Arguments:')
    print(args)
    print('---' * 30)

    '''Create and initialize dataset manager'''
    t_s = time.time()
    print('Create new behavior dataset manager:')
    dataset_manager = BehaviorDatasetManager()
    dataset_manager.read_itemlist_file(itemlist_file=args.itemlist)
    dataset_manager.read_behaviorlist_file(behaviorlist_file=args.behaviorlist)
    print('---' * 30)

    '''Create mtie model object'''
    print('Create MTIE model computation graph in Tensorflow:')
    args.number_of_items = dataset_manager.number_of_items #在构建graph的时候会用这个参数
    model_mtie = ModelMTIE(args)
    t_e = time.time()
    initialization_time = (t_e - t_s)

    '''Configurations for Tensorflow session'''
    sess_config = tf.ConfigProto(intra_op_parallelism_threads=args.threads,
                                 inter_op_parallelism_threads=args.threads)

    '''Start Tensorflow session'''
    with tf.Session(config=sess_config) as sess:
        print('---' * 30)
        print('Fire up TensorFlow session')

        '''Initialize variables'''
        tf.global_variables_initializer().run()

        curr_learning_rate = args.rho

        sampling_cum_time = 0
        training_cum_time = 0

        samples_number = args.samples * 1000

        # Report progress every 0.1% checkpoint
        checkpoint = samples_number // 1000



        summary_writer = tf.summary.FileWriter('snapshots/', sess.graph) #日志记录器



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
            feed_dict = {model_mtie.batch_behaviors_item_indices: batch_behaviors_item_indices,
                         model_mtie.batch_behaviors_item_type_indices: batch_behaviors_item_type_indices,
                         model_mtie.batch_behaviors_labels: batch_behaviors_labels,
                         model_mtie.learning_rate: curr_learning_rate}

            t_s = time.time()

            # sess.run(model_mtie.train_op, feed_dict=feed_dict)

            _, l1_loss_, merge_op , l1_lables, l1_encoded_, l1_product, l1_rates, l1_norms= sess.run([model_mtie.train_op, model_mtie.loss,model_mtie.merge_op,model_mtie.batch_behaviors_labels,model_mtie.batch_behaviors_weighted_representations,model_mtie.batch_behaviors_matrix_inner_product,model_mtie.batch_behaviors_success_rates,model_mtie.batch_behaviors_norms], feed_dict=feed_dict)


            summary_writer.add_summary(merge_op, sample_index)
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
            t_e = time.time()
            training_cum_time += (t_e - t_s)

            '''Report progress'''
            if not sample_index % checkpoint:
                avg_loop_time = (sampling_cum_time + training_cum_time) / (sample_index + 1)
                etc = (samples_number - sample_index) * avg_loop_time
                progress = sample_index / samples_number
                print('loss:{:.4f};\tCurrent rho: {:9.5f};\tProgress: {:7.1%};\tETC: {:5.1f} min'.
                      format(l1_loss_, curr_learning_rate, progress, (etc / 60)), end='\r\n', flush=True)

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
        target_embeddings = sess.run(model_mtie.target_item_embeddings)
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


if __name__ == '__main__':
    seed = 666
    random.seed(seed)
    np.random.seed(seed)

    train_mtie(parse_args())

## Usage
'''
python model_mtie.py --itemlist data/itemlist.txt --behaviorlist data/behaviorlist.txt --output embeddings.mode1 --mode 1 --threads 8

'''