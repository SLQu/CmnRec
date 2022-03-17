#!/usr/bin/env python
# coding:utf-8


# https://gaussic.github.io/2017/08/24/tensorflow-language-model/
from collections import Counter
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
import os
import numpy as np
import tensorflow as tf
import memory_uw
log_2 = np.log(2.0)
class PTBModel(object):
    def __init__(self,config,reuse=False):
        self.config = config
        self.reuse = reuse
        self.log = config.log
        self.num_steps=config.num_steps
        self.vocab_size=config.vocab_size
        self.batch_size = config.batch_size
        self.use_men = config.use_mem

        self.embedding_dim=config.embedding_dim
        self.hidden_dim=config.hidden_dim
        self.num_layers=config.num_layers
        self.rnn_model=config.rnn_model
        self.learning_rate=config.learning_rate
        self.use_cache = config.use_cache
        self.cache_attend_dim = config.cache_attend_dim

        self.cost_type = config.cost_type
        self.cost_fun = config.cost_fun

    def placeholders(self):
        self.wholesession = tf.placeholder('int32',
                                           [None, None], name='wholesession')
        self._inputs = self.wholesession[:, 0:-1]
        if self.cost_type == "all":
            self._targets = self.wholesession[:,1:]
        else:
            self._targets = self.wholesession[:, -1:]

        self.dropout_keep_prob = tf.placeholder(tf.float32)


    def build_model(self):

        self.placeholders()
        if self.config.time_type==0:
            if self.use_men == 1 & self.use_cache == 1:
                self.rnn()
            else:
                self.rnn_tf()
        # elif self.use_men == 1 & self.use_cache == 1:
        #     self.rnn_tf()
        else:
            self.rnn()

        self.cost()
        self.optimize()

    def input_embedding(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input_embedding') as scope:
                if self.reuse:
                    scope.reuse_variables()
                self.embedding=tf.get_variable("embedding",[self.vocab_size,self.embedding_dim],dtype=tf.float32)
                _inputs=tf.nn.embedding_lookup(self.embedding,self._inputs)
        return _inputs

    def rnn_tf(self):
        def lstm_cell():
            # return tf.contrib.rnn.BasicLSTM(self.hidden_dim,state_is_tuple=True)
            return tf.nn.rnn_cell.LSTMCell(self.hidden_dim,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_dim)
        def dropout_cell():
            if(self.rnn_model=='lstm'):
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout_keep_prob)

        def _loop_body(times,out,controller_state,_inputs):
            self.log.info('No memory')
            self.log.info('========================================================')
            output, controller_state = rnn_cell(_inputs[:, times, :], controller_state)
            times +=1
            # scope.reuse_variables()
            return times,out.write(times-1, output),controller_state,_inputs

        def _loop_body_men(times,outs,controller_state,_inputs,memory_state,cache_hiddens,times2):
            self.log.info('use memory')
            self.log.info('========================================================')
            last_read_vectors = memory_state[6]  # read values from memory


            flat_read_vectors = tf.reshape(last_read_vectors,(self.batch_size, -1))  # flatten R read vectors: batch x RN
            complete_input = tf.concat([_inputs[:, times, :], flat_read_vectors], 1)  # concat input --> read data
            nn_output, controller_state = rnn_cell(complete_input, controller_state)

            pre_output = tf.matmul(nn_output,self.memory.nn_output_weights)  # batch x output_dim -->later combine with new read vector

            if self.use_cache==1:

                self.log.info('use cache')
                def updateCache():

                    def getatt():
                        cache_values = cache_hiddens.gather(tf.range(times - times2, times + 1))
                        cache_values = tf.transpose(cache_values, [1, 0, 2])  # bs x Lin x h
                        U = tf.reshape(tf.matmul(tf.reshape(cache_values, [-1, self.hidden_dim]), self.memory.cU_a),
                                       [self.batch_size, -1, self.cache_attend_dim])
                        V = tf.reshape(
                            tf.matmul(tf.reshape(last_read_vectors,
                                                 [self.batch_size, self.memory.read_heads * self.memory.word_size]),
                                      self.memory.cV_a),
                            [self.batch_size, 1, self.cache_attend_dim])
                      #  H = tf.reshape(tf.matmul(nn_output, self.memory.cW_a), [self.batch_size, 1, self.cache_attend_dim])
                        total = U + V # + H
                        total = tf.reshape(tf.tanh(total), [-1, self.cache_attend_dim])

                        eijs = tf.matmul(total, tf.expand_dims(self.memory.cv_a, 1))  # bs.Lin x 1
                        eijs = tf.reshape(eijs, [self.batch_size, -1])  # bs x Lin
                        alphas = tf.nn.softmax(eijs)

                        att = tf.reduce_sum(cache_values * tf.expand_dims(alphas, 2), 1)  # bs x h x 1
                        return att

                    def cacheHidden():
                        return cache_hiddens.gather(tf.range(times - times2 , times + 1))

                    att = tf.cond(tf.equal(0,times2),cacheHidden,getatt)
                    att = tf.reshape(att, [self.batch_size, self.hidden_dim])  # bs x h
                    att_state = list(controller_state)

                    if self.config.rnn_model =="lstm" :
                        att_state[-1] = LSTMStateTuple(att_state[-1][0],att)
                    elif self.config.rnn_model =="gru":
                        att_state[-1]=att

                    att_state = tuple(att_state)
                    return 0,att_state

                def holdCache():
                    return times2+1,controller_state

                def updateMem():
                    interface = tf.matmul(nn_output, self.memory.interface_weights)  # batch x interface_dim
                    interface = tf.nn.dropout(interface,keep_prob=self.dropout_keep_prob)
                    interface = self.memory.parse_interface_vector(interface)  # use to read write into vector

                    usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = \
                        self.memory.write(
                            memory_state[0], memory_state[1], memory_state[5],
                            memory_state[4], memory_state[2], memory_state[3],
                            interface['write_key'],
                            interface['write_strength'],
                            interface['free_gates'],
                            interface['allocation_gate'],
                            interface['write_gate'],
                            interface['write_vector'],
                            interface['erase_vector']
                        )

                    read_weightings, new_read_vectors = self.memory.read(
                        memory_matrix,
                        memory_state[5],
                        interface['read_keys'],
                        interface['read_strengths'],
                        link_matrix,
                        interface['read_modes'],
                    )

                    new_memory_state = tuple([memory_matrix, usage_vector,
                                          precedence_vector, link_matrix,
                                          write_weighting, read_weightings, new_read_vectors])
                    return new_memory_state,new_read_vectors
                def holdMem():
                    return memory_state,last_read_vectors

                def judgeWrite():
                    sss = tf.cast(tf.equal(self.istimes, self.num_steps -1 - times), dtype=tf.int32)
                    ssaa = tf.reduce_sum(sss)
                    return tf.cast(ssaa, dtype=tf.bool)

                cache_hiddens = cache_hiddens.write(times, nn_output)
                times2, controller_state = tf.cond(judgeWrite(), updateCache, holdCache)
                ''' update memory  '''
                memory_state,read_vectors = tf.cond(tf.equal(times2, 0), updateMem, holdMem)

            else:
                interface = tf.matmul(nn_output, self.memory.interface_weights)  # batch x interface_dim
                interface = tf.nn.dropout(interface, keep_prob=self.dropout_keep_prob)
                interface = self.memory.parse_interface_vector(interface)  # use to read write into vector
                self.log.info('no cache')
                usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = \
                    self.memory.write(
                        memory_state[0], memory_state[1], memory_state[5],
                        memory_state[4], memory_state[2], memory_state[3],
                        interface['write_key'],
                        interface['write_strength'],
                        interface['free_gates'],
                        interface['allocation_gate'],
                        interface['write_gate'],
                        interface['write_vector'],
                        interface['erase_vector']
                    )
                read_weightings, read_vectors = self.memory.read(
                    memory_matrix,
                    memory_state[5],
                    interface['read_keys'],
                    interface['read_strengths'],
                    link_matrix,
                    interface['read_modes'],
                )
                memory_state = tuple([memory_matrix,usage_vector,
                                          precedence_vector,link_matrix,
                                          write_weighting,read_weightings,read_vectors])

            flat_read_vectors = tf.reshape(read_vectors, (self.batch_size, -1))  # batch_size x flatten
            final_output = pre_output + tf.matmul(flat_read_vectors, self.memory.mem_output_weights)

            times +=1
            return times,outs.write(times-1, final_output),\
                   controller_state,_inputs,memory_state,cache_hiddens,times2

        with tf.variable_scope('LSTM') as scope:
            lstm = [dropout_cell() for _ in range(self.num_layers)] #tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            rnn_cell = tf.contrib.rnn.MultiRNNCell(lstm, state_is_tuple=True)

            controller_state = rnn_cell.zero_state(self.batch_size, tf.float32)
            _inputs = self.input_embedding()
            outs = tf.TensorArray(dtype=tf.float32,size=self.num_steps)

            with tf.variable_scope("sequence__loop") as scope:
                if self.reuse:
                    scope.reuse_variables()

                times = tf.constant(0, dtype=tf.int32)
                times2 = tf.constant(0, dtype=tf.int32)
                if self.use_men:

                    self.memory = memory_uw.Memory(self.config.m_input_size,self.config.m_output_size,
                                                   self.config.words_num, self.config.word_size,
                                                   self.config.read_heads,self.batch_size)
                    memory_state = self.memory.init_memory()

                    if self.use_cache:
                        self.istimes = self.getWriteTimes(self.config.cache_type,True)
                        self.memory.ini_W(self.cache_attend_dim,self.hidden_dim)
                    cache_hiddens = tf.TensorArray(tf.float32,size=self.num_steps,dynamic_size=True,element_shape=[self.batch_size,self.hidden_dim])

                    _results = tf.while_loop(
                        cond=lambda times, *_: times < self.num_steps,
                        body=_loop_body_men,
                        loop_vars=(
                            times,outs,controller_state,_inputs,memory_state,cache_hiddens,times2
                        ),
                    )

                else:
                    _results = tf.while_loop(
                        cond=lambda times, *_: times < self.num_steps,
                        body=_loop_body,
                        loop_vars=(
                            times,outs,controller_state,_inputs
                        ),
                    )

                _results =tf.transpose(_results[1].stack(), [1, 0, 2])
            if self.cost_type == "all":
                last = _results
            else:
                last=_results[:,-1,:]
        self.hidden_out = last
        logits=tf.layers.dense(inputs=last,units=self.vocab_size,name="dense_layer",reuse=tf.AUTO_REUSE)
        self._logits=logits


        if self.cost_type == "all":
            self.input_y = tf.reshape(self._targets, [self.batch_size,self.num_steps])  # fajie addd
        else:
            self.input_y = tf.reshape(self._targets, [-1])  # fajie addd


    def rnn(self):
        def lstm_cell():
            # return tf.contrib.rnn.BasicLSTM(self.hidden_dim,state_is_tuple=True)
            return tf.nn.rnn_cell.LSTMCell(self.hidden_dim,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_dim)
        def dropout_cell():
            if(self.rnn_model=='lstm'):
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout_keep_prob)

        with tf.variable_scope('LSTM') as scope:
            lstm = [dropout_cell() for _ in range(self.num_layers)] #tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            rnn_cell = tf.contrib.rnn.MultiRNNCell(lstm, state_is_tuple=True)
            controller_state = rnn_cell.zero_state(self.batch_size, tf.float32)
            _inputs = self.input_embedding()

            with tf.variable_scope("sequence__loop") as scope:
                if self.reuse:
                    scope.reuse_variables()

                _results = []
                if self.use_men:
                    self.memory = memory_uw.Memory(self.config.m_input_size,self.config.m_output_size,self.config.words_num, self.config.word_size,self.config.read_heads,self.batch_size)
                    memory_state = self.memory.init_memory()
                    if self.use_cache:
                        self.timeToWrite = self.getWriteTimes(self.config.cache_type)
                        self.memory.ini_W(self.cache_attend_dim,self.hidden_dim)
                    cache_hiddens = []
                    times2 = 0

                    for times in range(self.num_steps):
                        self.log.info('use memory  '+str(times))
                        last_read_vectors = memory_state[6]  # read values from memory
                        flat_read_vectors = tf.reshape(last_read_vectors,(self.batch_size, -1))  # flatten R read vectors: batch x RN
                        complete_input = tf.concat([_inputs[:, times, :], flat_read_vectors],1)  # concat input --> read data
                        nn_output, controller_state = rnn_cell(complete_input, controller_state)
                        pre_output = tf.matmul(nn_output,self.memory.nn_output_weights)  # batch x output_dim -->later combine with new read vector

                        if self.use_cache == 1:
                            self.log.info('use cache')
                            cache_hiddens.append(nn_output)

                            if (self.num_steps - 1 - times) not in self.timeToWrite:
                                read_vectors = last_read_vectors
                                times2 +=1
                            else:
                                controller_state = self.updateCache(times, times2, cache_hiddens,last_read_vectors,controller_state)
                                memory_state, read_vectors = self.updateMem(nn_output,memory_state)
                                times2 = 0

                        else:
                            interface = tf.matmul(nn_output, self.memory.interface_weights)  # batch x interface_dim
                            interface = tf.nn.dropout(interface, keep_prob=self.dropout_keep_prob)
                            interface = self.memory.parse_interface_vector(interface)  # use to read write into vector
                            self.log.info('no cache')
                            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector =self.memory.write(memory_state[0], memory_state[1], memory_state[5],memory_state[4], memory_state[2], memory_state[3],interface['write_key'],interface['write_strength'],interface['free_gates'],interface['allocation_gate'],interface['write_gate'],interface['write_vector'],interface['erase_vector'])
                            read_weightings, read_vectors = self.memory.read(memory_matrix,memory_state[5],interface['read_keys'],interface['read_strengths'],link_matrix,interface['read_modes'],)
                            memory_state = tuple([memory_matrix, usage_vector,precedence_vector, link_matrix,write_weighting, read_weightings, read_vectors])

                        flat_read_vectors = tf.reshape(read_vectors, (self.batch_size, -1))  # batch_size x flatten
                        final_output = pre_output + tf.matmul(flat_read_vectors, self.memory.mem_output_weights)
                        _results.append(final_output)
                else:
                    for times in range(self.num_steps):
                        self.log.info('No memory')
                        print(_inputs[:, times, :])
                        print(controller_state)
                        output, controller_state = rnn_cell(_inputs[:, times , :], controller_state)
                        _results.append(output)

                _results =tf.transpose(tf.stack(_results), [1, 0, 2])
            if self.cost_type == "all":
                last = _results
            else:
                last=_results[:,-1,:]
        self.hidden_out = last
        logits=tf.layers.dense(inputs=last,units=self.vocab_size,name="dense_layer",reuse=tf.AUTO_REUSE)
        self._logits=logits

        if self.cost_type == "all":
            self.input_y = tf.reshape(self._targets, [self.batch_size,self.num_steps])  # fajie addd
        else:
            self.input_y = tf.reshape(self._targets, [-1])  # fajie addd


    def updateCache(self,times, times2, cache_hiddens,last_read_vectors,controller_state):

        if times2 != 0:
            cache_values = cache_hiddens[(times - times2): times + 1]
            cache_values = tf.transpose(cache_values, [1, 0, 2])  # bs x Lin x h
            U = tf.reshape(
                tf.matmul(tf.reshape(cache_values, [-1, self.hidden_dim]), self.memory.cU_a),
                [self.batch_size, -1, self.cache_attend_dim])
            V = tf.reshape(tf.matmul(tf.reshape(last_read_vectors, [self.batch_size,
                                                                    self.memory.read_heads * self.memory.word_size]),
                                     self.memory.cV_a),
                           [self.batch_size, 1, self.cache_attend_dim])
            total = U + V  # + H
            total = tf.reshape(tf.tanh(total), [-1, self.cache_attend_dim])
            eijs = tf.matmul(total, tf.expand_dims(self.memory.cv_a, 1))  # bs.Lin x 1
            eijs = tf.reshape(eijs, [self.batch_size, -1])  # bs x Lin
            alphas = tf.nn.softmax(eijs)
            att = tf.reduce_sum(cache_values * tf.expand_dims(alphas, 2), 1)  # bs x h x 1
            print("=================2==========================")
        else:
            att = cache_hiddens[-1]
            print("==================3=========================")

        att = tf.reshape(att, [self.batch_size, self.hidden_dim])  # bs x h
        att_state = list(controller_state)

        if self.config.rnn_model == "lstm":
            att_state[-1] = LSTMStateTuple(att_state[-1][0], att)
        elif self.config.rnn_model == "gru":
            att_state[-1] = att

        att_state = tuple(att_state)
        return att_state

    def updateMem(self,nn_output,memory_state):
        interface = tf.matmul(nn_output, self.memory.interface_weights)  # batch x interface_dim
        interface = tf.nn.dropout(interface, keep_prob=self.dropout_keep_prob)
        interface = self.memory.parse_interface_vector(
            interface)  # use to read write into vector

        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = \
            self.memory.write(
                memory_state[0], memory_state[1], memory_state[5],
                memory_state[4], memory_state[2], memory_state[3],
                interface['write_key'],
                interface['write_strength'],
                interface['free_gates'],
                interface['allocation_gate'],
                interface['write_gate'],
                interface['write_vector'],
                interface['erase_vector']
            )

        read_weightings, new_read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )

        new_memory_state = tuple([memory_matrix, usage_vector,
                                  precedence_vector, link_matrix,
                                  write_weighting, read_weightings, new_read_vectors])
        return new_memory_state, new_read_vectors

    def getWriteTimes(self,cache_type,while_l=False):
        times_num = int(self.num_steps / (1 + self.memory.words_num) / self.memory.words_num * 2)
        times_num_2 = int(self.num_steps / self.memory.words_num)
        if cache_type == 1:
            if times_num == 0:
                istimes = [i * times_num_2 for i in range(0, self.memory.words_num)]
                self.log.info("1   times_num = 0")
            else:
                istimes = [sum(range(i)) * times_num for i in range(1, self.memory.words_num + 1)]
                self.log.info("2   times_num = " + str(times_num))
        elif cache_type == 2:
            istimes = [i * times_num_2 for i in range(0, self.memory.words_num)]
        else:
            istimes = [i for i in range(0, self.memory.words_num)]

        if while_l:
            return tf.constant(istimes,dtype=tf.int32)
        else:
            return set(istimes)

    def evaluate(self):
        if self.cost_type == "all":
            if self.cost_fun == "bpr":
                logits = self.hidden_out[:, -1, :]
                logits = tf.matmul(logits, self.softmax_W, transpose_b=True) + self.softmax_b
            else:
                logits = self._logits[:, -1, :]
            _targets = self._targets[:, -1:]
        else:
            if self.cost_fun == "bpr":
                logits = self.hidden_out
                logits = tf.matmul(logits, self.softmax_W, transpose_b=True) + self.softmax_b
            else:
                logits = self._logits
            _targets = tf.reshape(self._targets,shape=[self.batch_size,-1])

        prediction = tf.nn.softmax(logits)

        lable = tf.reshape(_targets, shape=[self.batch_size])
        _,pred_words_5 = tf.nn.top_k(prediction, 5)
        bool_idx_5 = tf.equal(pred_words_5, tf.cast(_targets,dtype=tf.int32))
        int_index_5 = tf.cast(tf.where(bool_idx_5)[:,1],dtype=tf.float32)
        MRR_5 = tf.reduce_sum(1.0 / (int_index_5 + 1))/self.batch_size
        Rec_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prediction, lable, 5),tf.float32))
        ndcg_5 = tf.reduce_sum(1.0 / tf.log(int_index_5 + 2)) /self.batch_size/log_2

        _,pred_words_20 = tf.nn.top_k(prediction, 20)
        bool_idx_20 = tf.equal(pred_words_20, tf.cast(_targets,dtype=tf.int32))
        int_index_20 = tf.cast(tf.where(bool_idx_20)[:,1],dtype=tf.float32)
        MRR_20 = tf.reduce_sum(1.0 / (int_index_20 + 1))/self.batch_size
        print(prediction)
        print(lable)
        Rec_20 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prediction, lable, 20),tf.float32))
        ndcg_20 = tf.reduce_sum(1.0 / tf.log(int_index_20 + 2)) /self.batch_size/log_2

        return MRR_5,Rec_5,ndcg_5,MRR_20,Rec_20,ndcg_20

    def cost(self):
        if self.cost_fun == "bpr":
            self.softmax_W = tf.get_variable('softmax_w', [self.vocab_size, self.hidden_dim],
                                             initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            self.softmax_b = tf.get_variable('softmax_b', [self.vocab_size], initializer=tf.constant_initializer(0.0))

            logits_2D = tf.reshape(self.hidden_out, [-1, self.hidden_dim])

            label_flat = tf.reshape(self.input_y, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(self.vocab_size/10)  # sample 20% as negatives
            # tf.nn.nce_loss
            cost = tf.nn.sampled_softmax_loss(self.softmax_W, self.softmax_b, inputs = logits_2D,
                                              num_sampled = num_sampled,num_classes = self.vocab_size,labels=label_flat,)
        else:
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self.input_y)
        cost = tf.reduce_mean(cost)

        self.loss = cost

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,)
        self.optim = optimizer.minimize(self.loss)

