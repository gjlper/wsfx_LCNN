class Config(object):
    """CNN配置参数"""

    embedding_dim = 128  # 词向量维度
    seq_length_1 = 30  # 序列长度
    seq_length_2 = 30  # 序列长度
    num_classes = 2  # 类别数
    H = 5 #head number

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏层神经元


    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 200  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

class con1dConfig(object):
    filters = 1
    kernel_size = 5
