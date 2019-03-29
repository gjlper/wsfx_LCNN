#该模型是使用word2vec表示输入输出
# coding: utf-8

import tensorflow as tf
import code.models.config as config



class mlpmodel(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x_1 = tf.placeholder(tf.float32, [None, self.config.seq_length_1, self.config.embedding_dim], name='input_x_1')
        self.input_x_2 = tf.placeholder(tf.float32, [None, self.config.seq_length_2, self.config.embedding_dim], name='input_x_2')
        self.input_x_polar = tf.placeholder(tf.float32, [None, self.config.polar_classes], name='x_polar')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.mlp()

    def mlp(self):
        inputx1_mean = tf.reduce_mean(self.input_x_1,axis=1)
        inputx2_mean = tf.reduce_mean(self.input_x_2,axis=2)
        concat = tf.concat([self.input_x_polar, inputx1_mean, inputx2_mean],axis=-1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(concat, self.config.hidden_dim, name='fc1')#w*input+b,其中可以在此方法中指定w,b的初始值，或者通过tf.get_varable指定
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)#根据比例keep_prob输出输入数据，最终返回一个张量
            fc = tf.nn.relu(fc)#激活函数，此时fc的维度是hidden_dim

            # 关系分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')#将fc从[batch_size,hidden_dim]映射到[batch_size,num_class]输出
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

            #极性分类器
            self.polar_logits = tf.layers.dense(fc, self.config.polar_classes, name= 'fc3')
            self.polar_pred_cls = tf.argmax(tf.nn.softmax(self.polar_logits), 1)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy_y = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)#对logits进行softmax操作后，做交叉墒，输出的是一个向量
            cross_entropy_polar = tf.nn.softmax_cross_entropy_with_logits(logits=self.polar_logits, labels=self.input_x_polar)
            self.loss = tf.reduce_mean(cross_entropy_y) + tf.reduce_mean(cross_entropy_polar)#将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)#由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
