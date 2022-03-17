#!/usr/bin/env python
# coding:utf-8

import platform
import os
import tensorflow as tf
import numpy as np


def setArgs(args):
    if os.path.exists("config/"+args.c+".txt"):
        cons = list(open("config/" + args.c + ".txt", "r").readlines())
        for con in cons:
            con = con.strip('\n').split("=")
            if len(con) != 2:
                break
            setattr(args, con[0], con[1])
        args = argsRecheck(args)

    return  args

def printargs(args):
    log = args.log
    for arg in vars(args):
        log.info('%s:--%s', arg, getattr(args, arg))
    log.info("------------------------------------------------------")
    log.info("num_steps :"+str(args.num_steps)+", text_dir :"+str(args.text_dir))
    log.info("memory :"+str(args.use_mem)+", cache :"+str(args.use_cache)+", words_num :"+str(args.words_num)+", read_heads :"+str(args.read_heads) )
    log.info("======================================================")

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("---  There is this folder!  ---")

def elapsed(sec):
    if sec<60:
        return str(sec)[:4] + " sec"
    elif sec<(60*60):
        return str(sec/60)[:4] + " min"
    else:
        return str(sec/(60*60))[:4] + " hr"

def getlog(log_name):
    # get TF logger
    import logging
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    mkdir('./log/')
    # create file handler which logs even debug messages
    fh = logging.FileHandler('./log/'+log_name+'.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log

def argsRecheck(args):
    args.learning_rate = float(args.learning_rate)
    args.sample_percentage = float(args.sample_percentage)
    args.use_mem = int(args.use_mem)
    args.use_cache = int(args.use_cache)
    args.words_num = int(args.words_num)
    args.dataset = int(args.dataset)
    args.read_heads = int(args.read_heads)
    args.batch_size = int(args.batch_size)
    args.embedding_dim = int(args.embedding_dim)
    args.hidden_dim = int(args.hidden_dim)
    args.cache_attend_dim = int(args.cache_attend_dim)

    args.word_size = int(args.word_size)
    args.top_k = int(args.top_k)
    args.max_epochs = int(args.max_epochs)
    args.num_layers = int(args.num_layers)

    return args



def creatCos(args,total_sess,embed,session):
    se_len, setp_len = total_sess.shape
    item_num, embed_size = embed.shape
    print(se_len, setp_len)
    print(item_num, embed_size)

    embedding = tf.convert_to_tensor(embed)

    sess_input = tf.placeholder('int32',[None, None])

    sess_embed_1 = tf.nn.embedding_lookup(embedding,sess_input[:,:-1])
    sess_embed_last = tf.nn.embedding_lookup(embedding,sess_input[:,-1])
    cons_1 = tf.matmul(sess_embed_1, tf.expand_dims(sess_embed_last, 2))
    cons_1 = tf.squeeze(cons_1, 2)

    sess_embed_last_2 = tf.square(sess_embed_last)
    sess_embed_1_2 = tf.square(sess_embed_1)

    sess_embed_last_2_red = tf.reduce_sum(sess_embed_last_2, 1)
    sess_embed_1_2_red = tf.reduce_sum(sess_embed_1_2, 2)
    sess_embed_last_2_qsrt = tf.sqrt(sess_embed_last_2_red)
    sess_embed_1_2_qsrt = tf.sqrt(sess_embed_1_2_red)

    a = tf.constant(np.ones(setp_len - 1), dtype=tf.float32)
    b = tf.transpose(sess_embed_last_2_qsrt)

    dowm_temp = tf.matmul(tf.expand_dims(b, 1), tf.expand_dims(a, 0))
    down = sess_embed_1_2_qsrt * dowm_temp

    out = tf.reduce_mean(cons_1 / down, axis=0)

    args.log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    batch_size = args.batch_size
    coss=np.zeros(shape=[setp_len-1],dtype=float)
    count = 0
    for i in range(int(se_len/batch_size)):
        sess_in_id = total_sess[i * batch_size: (i + 1) * batch_size,:]
        aa = session.run(out,feed_dict={sess_input: sess_in_id})
        coss+=np.array(aa)
        count+=1

    args.log.info(coss/count)

def creatSessonCos(args,total_sess,embed,session):
    se_len, setp_len = total_sess.shape
    item_num, embed_size = embed.shape
    print(se_len, setp_len)
    print(item_num, embed_size)

    embedding = tf.convert_to_tensor(embed)
    sess_input = tf.placeholder('int32',[None, None])
    sess_embed_one = tf.nn.embedding_lookup(embedding,sess_input[:,:-1])
    sess_embed_two = tf.nn.embedding_lookup(embedding,sess_input[:,1:])
    aa = tf.reshape(sess_embed_one,shape = [-1,embed_size])
    bb = tf.reshape(sess_embed_two,shape = [-1,embed_size])

    cons_u = tf.reduce_sum(tf.multiply(aa,bb),1)
    cons_up = tf.reshape(cons_u,shape=[-1,setp_len-1])


    sess_embed_one_d = tf.square(sess_embed_one)
    sess_embed_two_d = tf.square(sess_embed_two)

    sess_embed_one_d = tf.reduce_sum(sess_embed_one_d, 2)
    sess_embed_two_d = tf.reduce_sum(sess_embed_two_d, 2)
    sess_embed_one_d = tf.sqrt(sess_embed_one_d)
    sess_embed_two_d = tf.sqrt(sess_embed_two_d)


    cos_dowm_ = tf.multiply(sess_embed_one_d,sess_embed_two_d)

    out = tf.reduce_mean(cons_up / cos_dowm_, axis=0)

    args.log.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    batch_size = args.batch_size
    coss=np.zeros(shape=[setp_len-1],dtype=float)
    count = 0
    for i in range(int(se_len/batch_size)):
        sess_in_id = total_sess[i * batch_size: (i + 1) * batch_size,:]
        aa = session.run(out,feed_dict={sess_input: sess_in_id})
        coss+=np.array(aa)
        count+=1

    args.log.info(coss/count)


def sample_top(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    probs = a[idx]
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice

# fajie
def sample_top_k(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    # probs = a[idx]
    # probs = probs / np.sum(probs)
    # choice = np.random.choice(idx, p=probs)
    return idx

# print(sample_top_k(np.array([0.02,0.01,0.01,0.16,0.8]),3))


