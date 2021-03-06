# coding: utf-8

import sys
import tensorflow.contrib.keras as kr
from collections import Counter
import pandas as pd
import numpy as np

import numpy as np


if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def data_convert(vectors):
    ssls = list(filter(lambda x:x.strip() != '', vectors))
    n_vector = [list(map(float, list(filter(lambda x: x.strip() != '', ss.split('//'))))) for ss in ssls]
    return n_vector

#2 inputs
def data_load2(data_f,config):
    input_x1, input_x2,  input_y = [], [], []
    lines = data_f.read().split('\n')
    for i in range(len(lines)):
        line = lines[i]
        print('index:', i)
        if line.strip() == "":
            continue

        array = line.split('|')
        if len(array) < 5:
            continue
        ssls = array[1].split(' ')
        ftzwls = array[2].split(' ')
        label = int(array[3].strip())
        input_x1.append(data_convert(ssls))
        input_x2.append(data_convert(ftzwls))
        if label == 0:
            input_y.append([1, 0])
        else:
            input_y.append([0, 1])

    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.seq_length_1)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.seq_length_2)

    return train_1, train_2,  np.array(input_y)

#used with data_load2
def batch_iter2(x1, x2,  y, batch_size=128):
    """生成批次数据"""
    data_len = len(x1)
    num_batch = int(data_len / batch_size)

    indices = np.random.permutation(np.arange(data_len)) #洗牌
    x1_shuffle = x1[indices]
    x2_shuffle = x2[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_shuffle[start_id:end_id],x2_shuffle[start_id:end_id],  y_shuffle[start_id:end_id]

def batch_iter2_test(x1, x2, y, batch_size=128):
    """生成批次数据"""
    data_len = len(x1)
    num_batch = int(data_len / batch_size)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1[start_id:end_id], x2[start_id:end_id], y[start_id:end_id]


#3 inputs
def data_load(data_f, config):
    input_x1,input_x2,input_ks,input_y = [], [], [], []
    lines = data_f.read().split('\n')
    for i in range(len(lines)):
        line = lines[i]
        print('index:',i)
        if line.strip() == "":
            continue

        array = line.split('|')
        if len(array) < 5:
            continue
        ssls = array[1].split(' ')
        ftzwls = array[2].split(' ')
        label = int(array[3].strip())
        polar = int(array[4].strip())
        input_x1.append(data_convert(ssls))
        input_x2.append(data_convert(ftzwls))
        if label==0: input_y.append([1,0])
        else: input_y.append([0,1])
        zsvector = np.zeros(shape=[1,3])
        if polar != 0: zsvector[polar-1] = 1
        input_ks.append(list(zsvector))
        print(polar,zsvector)

    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.FACT_LEN)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.LAW_LEN)
    train_ks = np.array(input_ks)

    return train_1,train_2,train_ks,np.array(input_y)

def embedding_load(words_f):
    cpslist = words_f.read().split('\n')
    embedding_dict = {}
    for i in range(len(cpslist)): embedding_dict[cpslist[i].strip()] = i + 1
    embedding_dict['NOT_FOUND'] = len(cpslist) + 1
    embedding_dict['PAD'] = 0
    embedding = np.float32(np.random.uniform(-0.02, 0.02, [len(embedding_dict), 50]))
    return embedding



def batch_iter(x1, x2, ks, y, batch_size=128):
    """生成批次数据"""
    data_len = len(x1)
    num_batch = int(data_len / batch_size)

    indices = np.random.permutation(np.arange(data_len)) #洗牌
    x1_shuffle = x1[indices]
    x2_shuffle = x2[indices]
    ks_shuffle = ks[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_shuffle[start_id:end_id],x2_shuffle[start_id:end_id], ks_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def batch_iter_test(x1,x2,ks,y,batch_size=128):
    data_len = len(x1)
    num_batch = int(data_len / batch_size)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1[start_id:end_id], x2[start_id:end_id], ks[start_id:end_id], y[start_id:end_id]

