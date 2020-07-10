# encoding:utf-8
from keras.callbacks import ModelCheckpoint

from preprocess import *
from generator import bert_generator
import ModelLib
import config
import numpy as np
import os
import gc
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

para = config.para
train_x, train_y, val_x, val_y, word2id, tags = pickle.load(open(para["data_pk_path"], 'rb'))


def train_bert_model(para, use_generator=False):
    para['tag_num'] = len(tags)
    model = ModelLib.BERT_MODEL(para)
    checkpoint = ModelCheckpoint(para["model_path"], monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    if use_generator:
        val_bert = load_path_bert(para["test_path"], para["sep"])
        model.fit_generator(bert_generator(para["batch_size"], para["train_path"], para["sep"], train_y, Shuffle=True),
                            steps_per_epoch=int(train_y.shape[0] / para["batch_size"]) + 1, callbacks=[checkpoint],
                            validation_data=(val_bert, val_y), epochs=para["EPOCHS"])  # , verbose=1
    else:
        # @why 获取bert嵌入
        # train_bert, val_bert = load_bert_repre()
        train_bert_1 = np.load("./rmrb/train_x_1.npy")
        train_bert_2 = np.load("./rmrb/train_x_2.npy")
        train_bert_3 = np.load("./rmrb/train_x_3.npy")
        train_bert = np.concatenate([train_bert_1, train_bert_2, train_bert_3])
        del train_bert_1, train_bert_2, train_bert_3
        gc.collect()
        val_bert = np.load("./rmrb/val_x.npy")
        print("Load Data Done.")
        model.fit(train_bert, train_y, batch_size=para["batch_size"], epochs=para["EPOCHS"], callbacks=[checkpoint],
                  validation_data=(val_bert, val_y), shuffle=True)  # , verbose=1


if __name__ == "__main__":
    train_bert_model(para, use_generator=False)
