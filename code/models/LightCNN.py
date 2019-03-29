import tensorflow as tf
import code.models.config as config
class LightCNN:
    def __init__(self,config):
        self.config = config
        self.input_x_1 = tf.placeholder(tf.float32, [None, self.config.seq_length_1, self.config.embedding_dim], name='input_x_1')
        self.input_x_2 = tf.placeholder(tf.float32, [None, self.config.seq_length_2, self.config.embedding_dim], name='input_x_2')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.LightCNN()

    def lightcnn(self,input,index,softmax=True):
        with tf.name_scope("cnn"+str(index)):
            B, L, D = tf.get_shape(input)
            d = D / self.config.H
            split_x = tf.transpose(tf.reshape(input, shape=[B, L, d, self.config.H]), [3, 0, 2, 1])  #[H,B,d,L]
            output = tf.zeros_likes(split_x)
            with tf.variable_scope("cnn"+str(index),reuse=True):
                weight = tf.get_variable(name='filters',shape=[self.config.H, 1, 5],dtype=tf.float32,trainable=True,
                                         initializer=tf.random_normal_initializer())
                if softmax:
                    weight = tf.nn.softmax(weight,axis=-1)
                for i in range(self.config.H):
                    output[i] = tf.layers.conv1d(inputs=split_x[i], filters=1, kernel_size=5, strides=1, padding='SAME',
                                                 kernel_initializer=weight, name='conv' + str(i))

            return tf.reshape(output, shape=[B, L, -1])

    def LightCNN(self):
        output1 = self.lightcnn(self.input_x_1,index=1, softmax=True) #[B,L,D]
        output2 = self.lightcnn(self.input_x_2,index=2, softmax=True) #[B,L,D]
        gmp1 = tf.reduce_max(output1, reduction_indices=[1], name='gmp1')
        gmp2 = tf.reduce_max(output2, reduction_indices=[1], name='gmp2')

        with tf.name_scope("concat"):
            concat = tf.concat([gmp1, gmp2], 1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(concat, self.config.hidden_dim,
                                 name='fc1')  # w*input+b,其中可以在此方法中指定w,b的初始值，或者通过tf.get_varable指定
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)  # 根据比例keep_prob输出输入数据，最终返回一个张量
            fc = tf.nn.relu(fc)  # 激活函数，此时fc的维度是hidden_dim

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes,
                                          name='fc2')  # 将fc从[batch_size,hidden_dim]映射到[batch_size,num_class]输出
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.input_y)  # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
            self.loss = tf.reduce_mean(cross_entropy)  # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                                    self.y_pred_cls)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))







