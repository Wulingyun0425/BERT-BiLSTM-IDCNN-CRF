# encoding:utf-8
from keras.callbacks import ModelCheckpoint, Callback

from preprocess import *
from generator import bert_generator
import ModelLib
import config
import numpy as np
import os
import gc
import time
from test import pos_F1

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

para = config.para
train_x, train_y, val_x, val_y, word2id, tags = pickle.load(open(para["data_pk_path"], 'rb'))


def predict_bert(pred_y):
    para['tag_num'] = len(tags)
    lengths = get_lengths(val_x)

    tag_pred_y = []
    tag_val_y = []
    for i, y in enumerate(pred_y):
        y = [np.argmax(dim) for dim in y]
        # print(lengths[i])
        p_y = y[:lengths[i]]
        # print(p_y)
        v_y = val_y[i][:lengths[i]].flatten()
        # print(v_y)
        p_y = [tags[dim] for dim in p_y]
        v_y = [tags[dim] for dim in v_y]
        tag_pred_y.append(p_y)
        tag_val_y.append(v_y)
    return tag_pred_y, tag_val_y


class EvaluateCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        pred_y = self.model.predict(self.validation_data[0])
        pred_y, val_y_1 = predict_bert(pred_y)
        P, R, F = pos_F1(pred_y, val_y_1)
        print("epoch:\t" + str(epoch))
        print("P:\t" + str(P))
        print("R:\t" + str(R))
        print("F1:\t" + str(F))


def train_bert_model(para, use_generator=False):
    para['tag_num'] = len(tags)
    model = ModelLib.BERT_MODEL(para)
    checkpoint = ModelCheckpoint(para["model_path"], monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    if use_generator:
        val_bert = load_path_bert(para["test_path"], para["sep"])
        model.fit_generator(bert_generator(para["batch_size"], para["train_path"], para["sep"], train_y, Shuffle=True),
                            steps_per_epoch=int(train_y.shape[0] / para["batch_size"]) + 1,
                            callbacks=[checkpoint, EvaluateCallback()], validation_data=(val_bert, val_y),
                            epochs=para["EPOCHS"])  # , verbose=1
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
        model.fit(train_bert, train_y, batch_size=para["batch_size"], epochs=para["EPOCHS"], shuffle=True,
                  callbacks=[checkpoint, EvaluateCallback()], validation_data=(val_bert, val_y))  # , verbose=1


if __name__ == "__main__":
    train_bert_model(para, use_generator=False)
