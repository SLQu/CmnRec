#!/usr/bin/env python
# coding:utf-8

import os
import platform

print(os.getcwd())
import tensorflow as tf
import numpy as np
import Tools
import data_loader_recsys as data_loader
import time
import sys
from rnn import PTBModel

import argparse
import time

'''
reimplementation of
Session-based Recommendations with Recurrent Neural Networks
screen print has been changed a bit so that to print the output not that ofen
'''
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_mem', type=int, default=1,
                        help='use_mem')
    parser.add_argument('--use_cache', type=int, default=1,
                        help='use cache')
    parser.add_argument('--words_num', type=int, default=3,
                        help='words_num')
    parser.add_argument('--cache_type', type=int, default=1,
                        help='cache_type')
    parser.add_argument('--dataset_rate', type=float, default=1.0,
                        help='dataset_rate')

    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='hidden layer size')
    parser.add_argument('--word_size', type=int, default=512,
                        help='word size')

    parser.add_argument('--dataset', type=str, default="demo.csv",
                        help='Directory containing text files')
    parser.add_argument('--rnn_model', type=str,
                        default='lstm', help='gru, lstm')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--read_heads', type=int, default=3,
                        help='number read heads')
    parser.add_argument('--time_type', type=int, default=0,
                        help="time test")
    parser.add_argument('--write_sess_emd', type=int, default=0,
                        help="write_sess_emd")
    parser.add_argument('--cost_type', type=str, default="all",
                        help='last or all')
    parser.add_argument('--cost_fun', type=str, default="bpr",
                        help='bpr or cross')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Learning Rate')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='embedding size')
    parser.add_argument('--cache_attend_dim', type=int, default=32,
                        help='cache attend dim')

    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Max Epochs')

    parser.add_argument('--seed', type=str,
                        default='f78c95a8-9256-4757-9a9f-213df5c6854e,1151b040-8022-4965-96d2-8a4605ce456c',
                        help='Seed for text generation')
    parser.add_argument('--sample_percentage', type=float, default=0.2,
                        help='sample_percentage from whole data, e.g.0.2= 80% training 20% testing')

    # parser.add_argument('--l2_reg_lambda', type=float, default=0,
    #                     help='L2 regularization lambda (default: 0.0)')

    parser.add_argument('--num_layers', type=int, default=1,
                        help='num_layers')

    parser.add_argument('--log_name', type=str,
                        default=time.strftime("%Y%m%d-%H%M%S", time.localtime()),
                        help='log_name')
    parser.add_argument('-c', type=str,
                        default="__none", help='config name')

    args = parser.parse_args()

    log = Tool.getlog(args.log_name)
    args.log = log

    log.info(len(sys.argv))
    log.info(str(sys.argv))

    if len(sys.argv)==2:
        args.c = sys.argv[1].split("=")[1]
        args = Tool.setArgs(args)

    args.m_input_size = args.hidden_dim
    args.m_output_size = args.hidden_dim
    args.text_dir = args.dataset
    dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.text_dir,'dataset_rate':args.dataset_rate})
    # text_samples=16390600  vocab=947255  session100
    all_samples = dl.item
    items = dl.item_dict


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    text_samples = all_samples[shuffle_indices]
    if args.write_sess_emd:
        args.text_samples = text_samples

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(args.sample_percentage * float(len(text_samples)))
    x_train, x_dev = text_samples[:dev_sample_index], text_samples[dev_sample_index:]
    # print(x_train.shape)
    # print((x_train[:,30:]).shape)
    # x_train=np.hstack((np.zeros(shape=[x_train.shape[0],30],dtype=float),x_train[:,30:]))
    # print(x_train.shape)
    args.num_steps = x_train.shape[1]-1
    args.x_train = x_train
    args.x_dev = x_dev
    args.train_size = x_train.shape[0]
    args.test_size = x_dev.shape[0]
    args.vocab_size = len(items)

    Tool.printargs(args)

    rnn=PTBModel(args)
    rnn.build_model()
    # loss_sum = tf.summary.scalar('loss', rnn.loss)
    log.info("train build ok , next is test")

    MRR_5,Rec_5,ndcg_5,MRR_20,Rec_20,ndcg_20 = rnn.evaluate()

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

    batch_size = args.batch_size
    total_loss, acc_total, last_loss = [0, 0, 0]
    log.info('train  begin')
    for epoch in range(args.max_epochs):
        batch_no = 0
        start = time.time()
        args.sss = 2000
        args.ssss = 0
        while (batch_no + 1) * batch_size <  args.train_size:   #args.train_size
            # do not need to evaluate all, only after several 10 sample_every, then output final results
            text_batch = x_train[batch_no * batch_size: (batch_no + 1) * batch_size, :]

            _, loss = sess.run([rnn.optim, rnn.loss ],
                               feed_dict={
                                   rnn.wholesession: text_batch,
                                   rnn.dropout_keep_prob: 0.5
                               })
            total_loss +=loss
            batch_no += 1

        log.info("  EPOCH: %d  total_loss: %.6f    delta loss: %.6f ",epoch,total_loss,(total_loss - last_loss))
        train_time = Tool.elapsed(time.time() - start)
        last_loss = total_loss
        total_loss = 0

        batch_no_test = 0
        curr_preds_5,rec_preds_5,ndcg_preds_5,curr_preds_20,rec_preds_20,ndcg_preds_20 = [[],[],[],[],[],[]]
        start = time.time()
        while (batch_no_test + 1) * batch_size <  args.test_size:  #    test_size    sss
            text_batch = x_dev[batch_no_test * batch_size: (batch_no_test + 1) * batch_size, :]
            batch_no_test += 1

            MRR_5__,Rec_5__,ndcg_5__,MRR_20__,Rec_20__,ndcg_20__ = sess.run(
                [MRR_5,Rec_5,ndcg_5,MRR_20,Rec_20,ndcg_20],
                feed_dict={
                    rnn.wholesession: text_batch,
                    rnn.dropout_keep_prob: 1.0
                })

            curr_preds_5.append(MRR_5__)
            rec_preds_5.append(Rec_5__)  # 2
            ndcg_preds_5.append(ndcg_5__)  # 2
            curr_preds_20.append(MRR_20__)
            rec_preds_20.append(Rec_20__)  # 2
            ndcg_preds_20.append(ndcg_20__)  # 2



        log.info("  EPOCH: %d  train time: %s  , test time: %s ,  %s ",
                 epoch,train_time, Tool.elapsed(time.time() - start),time.strftime("%m%d-%H%M%S", time.localtime()))
        log.info("  <mrr_5,20,hit_5,20,ndcg_5,20> %.4f, %.4f, %.4f, %.4f, %.4f, %.4f",
                 sum(curr_preds_5) / batch_no_test,
                 sum(curr_preds_20) / batch_no_test,
                 sum(rec_preds_5) / batch_no_test,
                 sum(rec_preds_20) / batch_no_test,
                 sum(ndcg_preds_5) / batch_no_test,
                 sum(ndcg_preds_20) / batch_no_test)  # 5

        log.info("==================================================================")
        # if args.write_sess_emd:
        #     if epoch==0:
        #         continue
        #     elif epoch%15==0 :
        #         Tool.creatCos(args, text_samples, np.array(sess.run(rnn.embedding)), sess)
        # Tool.creatSessonCos(args, text_samples, np.array(sess.run(rnn.embedding)), sess)


if __name__ == '__main__':
    main()
