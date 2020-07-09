# encoding:utf-8

import numpy as np
from collections import Counter

from keras import Sequential
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
import config
import random
import pickle
import codecs
from bert_serving.client import BertClient

para = config.para


def get_char2id(train_x, id2id, maxlen):
    char_l = []
    lose = [0] * maxlen
    for sentence in train_x:
        sent_l = []
        for word_id in sentence:
            try:
                # print(id2id[word_id])
                sent_l.append(id2id[word_id])
            except Exception as e:
                sent_l.append(lose)
        char_l.append(sent_l)
    return char_l


def cross_validation(X, Y, fold):
    val_X = []
    val_Y = []
    train_X = []
    train_Y = []
    step = int(X.__len__() / fold)
    for i in range(fold):
        if i != fold - 1:
            val_X.append(X[step * i:step * (i + 1)])
            val_Y.append(Y[step * i:step * (i + 1)])
        else:
            val_X.append(X[step * i:])
            val_Y.append(Y[step * i:])
    for i in range(fold):
        X_list = []
        Y_list = []
        for j in range(val_X.__len__()):
            if j != i:
                X_list.append(val_X[j])
                Y_list.append(val_Y[j])
        train_X.append(np.concatenate(X_list, axis=0))
        train_Y.append(np.concatenate(Y_list, axis=0))
    return train_X, train_Y, val_X, val_Y


def train_test_dev_preprocess():
    train = _parse_data(codecs.open(para["train_path"], 'r'), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r'), sep=para["sep"])
    dev = _parse_data(codecs.open(para["dev_path"], 'r'), sep=para["sep"])
    # train_len = train.__len__()
    print("Load dataset finish!!")
    dataset = train + test + dev
    tags = get_tag(dataset)
    print(tags)
    print(train.__len__(), test.__len__(), dev.__len__(), dataset.__len__())
    word_counts = Counter(row[0].lower() for sample in dataset for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    word2id = dict((w, i + 1) for i, w in enumerate(vocab))

    train_X, train_Y = process_data(train, word2id, tags)
    dev_X, dev_Y = process_data(dev, word2id, tags)
    test_X, test_Y = process_data(test, word2id, tags)
    pickle.dump((train_X, train_Y, test_X, test_Y, dev_X, dev_Y, word2id, tags), open(para["data_pk_path"], "wb"))


def train_test_set_preprocess():
    # @why 读取构造数据集为list
    test = _parse_data(codecs.open(para["test_path"], 'r', encoding="UTF-8"), sep=para["sep"])
    train = _parse_data(codecs.open(para["train_path"], 'r', encoding="UTF-8"), sep=para["sep"])
    print("Load trainset,dataset finish!!")
    dataset = train + test
    # @why 获取标签数目
    tags = get_tag(dataset)
    print(tags)
    print(train.__len__(), test.__len__(), dataset.__len__())
    # @why 词频统计
    word_counts = Counter(row[0].lower() for sample in dataset for row in sample)
    # @why 取词频高的构成词表
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    # @why 构建word2id
    word2id = dict((w, i + 1) for i, w in enumerate(vocab))
    train_X, train_Y = process_data(train, word2id, tags)
    test_X, test_Y = process_data(test, word2id, tags)
    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)
    pickle.dump((train_X, train_Y, test_X, test_Y, word2id, tags), open(para["data_pk_path"], "wb"))


def load_bert_repre():
    train = _parse_data(codecs.open(para["train_path"], 'r', encoding="UTF-8"), sep=para["sep"])
    test = _parse_data(codecs.open(para["test_path"], 'r', encoding="UTF-8"), sep=para["sep"])
    # @why 取数据不带标签
    train = [[items[0] for items in sent] for sent in train]
    test = [[items[0] for items in sent] for sent in test]

    # train_x = np.zeros(shape=(train.__len__(), para["max_len"], 768), dtype="float32")
    # test_x = np.zeros(shape=(test.__len__(), para["max_len"], 768), dtype="float32")

    bc = BertClient()
    x = bc.encode(test[:], is_tokenized=True)
    x = x[:, 1:para["max_len"] + 1]
    print("save val_x")
    np.save("./rmrb/val_x", x)

    bc = BertClient()
    x = bc.encode(train[:5000], is_tokenized=True)
    x = x[:, 1:para["max_len"] + 1]
    np.save("./rmrb/train_x_1", x)
    print("save 5000")

    bc = BertClient()
    x = bc.encode(train[5000:10000], is_tokenized=True)
    x = x[:, 1:para["max_len"] + 1]
    np.save("./rmrb/train_x_2", x)
    print("save 10000")

    bc = BertClient()
    x = bc.encode(train[10000:15000], is_tokenized=True)
    x = x[:, 1:para["max_len"] + 1]
    np.save("./rmrb/train_x_3", x)
    print("save 15000")

    bc = BertClient()
    x = bc.encode(train[15000:], is_tokenized=True)
    x = x[:, 1:para["max_len"] + 1]
    np.save("./rmrb/train_x_4", x)
    print("save all")

    # step = int(train.__len__() / 256) + 1
    # for i in range(step):
    #     if i != step - 1:
    #         x = bc.encode(train[i * 256:(i + 1) * 256], is_tokenized=True)
    #         x = x[:, 1:para["max_len"] + 1]
    #         train_x[i * 256:((i + 1) * 256)] = x
    #         # print(train_x[i*256:(i+1)*256])
    #     else:
    #         x = bc.encode(train[i * 256:], is_tokenized=True)
    #         x = x[:, 1:para["max_len"] + 1]
    #         train_x[i * 256:] = x
    #         # print(train_x[i*256:])
    #
    # step = int(test.__len__() / 256) + 1
    # # print(step)
    # for i in range(step):
    #     if i != step - 1:
    #         x = bc.encode(test[i * 256:(i + 1) * 256], is_tokenized=True)
    #         x = x[:, 1:para["max_len"] + 1]
    #         test_x[i * 256:((i + 1) * 256)] = x
    #         print(test_x[i * 256:(i + 1) * 256])
    #     else:
    #         x = bc.encode(test[i * 256:], is_tokenized=True)
    #         x = x[:, 1:para["max_len"] + 1]
    #         test_x[i * 256:] = x
    #         # print(test_x[i * 256:])
    return None  # train_x, test_x


def load_path_bert(path, sep="\t"):
    test = _parse_data(codecs.open(path, 'r', encoding="UTF-8"), sep=sep)
    test = [[items[0] for items in sent] for sent in test]
    test_x = np.zeros(shape=(test.__len__(), para["max_len"], 768), dtype="float32")
    bc = BertClient()

    step = int(test.__len__() / 256) + 1
    print(step)
    for i in range(step):
        if i != step - 1:
            x = bc.encode(test[i * 256:(i + 1) * 256], is_tokenized=True)
            x = x[:, 1:para["max_len"] + 1]
            test_x[i * 256:((i + 1) * 256)] = x
            # print(test_x[i * 256:(i + 1) * 256])
        else:
            x = bc.encode(test[i * 256:], is_tokenized=True)
            x = x[:, 1:para["max_len"] + 1]
            test_x[i * 256:] = x
            # print(test_x[i * 256:])
    # pickle.dump(test_x, open("./data/bert-pku-seg.pk", "wb"))
    return test_x


def get_tag(data):
    tag = []
    for words in data:
        for word_tag in words:
            if word_tag[1] not in tag:
                tag.append(word_tag[1])
    return tag


def _parse_data(file_input, sep="\t"):
    biaodian = ["。", ",", "，", "!", "！", "?", "？", "、", "；"]
    rows = file_input.readlines()
    items = [row.strip().split(sep) for row in rows]
    sents = []
    sent = []
    n = 0
    for item in items:
        if len(item) > 1:
            sent.append(item)
        else:
            sents.append(sent)
            sent = []
    new_sents = []
    for sent in sents:
        if len(sent) <= 128:
            new_sents.append(sent)
        else:
            n += 1
    print("Discard " + str(n) + " sentences")
    return new_sents


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i + 1) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, maxlen, padding='post', truncating='post')  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1, padding='post', truncating='post')
    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
        # print(y_chunk)
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk, word2idx


def process_data(data, word2idx, chunk_tags, onehot=False):
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    # @why padding到max_len，在后面padding
    x = pad_sequences(x, para["max_len"], padding='post', truncating='post')  # left padding
    y_chunk = pad_sequences(y_chunk, para["max_len"], value=-1, padding='post', truncating='post')

    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
        # print(y_chunk)
    else:
        # @why 第三个维度加上
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def get_lengths(X):
    lengths = []
    for i in range(len(X)):
        length = 0
        for dim in X[i]:
            # print(dim)
            if dim != 0:
                length += 1
            else:
                break
        # print(length)
        lengths.append(length)

    return lengths


def create_bool_matrex(repre_dim, x):
    bool_x = np.zeros(shape=(x.shape[0], x.shape[1], repre_dim))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] != 0:
                bool_x[i, j, :] = 1.
    return bool_x


def load_embed_weight(word2id):
    embed_weight = np.zeros(shape=(len(word2id.keys()) + 1, para["embed_dim"]))
    char2vec = {}
    with open(para["embed_path"], "r") as f:
        rows = f.readlines()
        for row in rows:
            item = row.strip().split(" ", 1)
            char = item[0]
            # print(item)
            vec_str = item[1].split(" ")
            vec = [float(i) for i in vec_str]
            char2vec[char] = vec
    for word in word2id.keys():
        # print(word)
        vec = char2vec[word]
        embed_weight[word2id[word]] = np.array(vec)
    print(embed_weight)
    return embed_weight


def get_simple2traditional():
    simple2traditional = {}
    with open(config.traditional_dict_path, "r") as f:
        rows = f.readlines()
        for row in rows:
            item = row.strip().split("	")
            simple2traditional[item[0]] = item[1]
    return simple2traditional


if __name__ == "__main__":
    train_test_set_preprocess()
