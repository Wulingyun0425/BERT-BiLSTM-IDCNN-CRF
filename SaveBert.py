import config
import pickle
import numpy as np
from preprocess import load_bert_repre

para = config.para
train_x, train_y, val_x, val_y, word2id, tags = pickle.load(open(para["data_pk_path"], 'rb'))

# para['tag_num'] = len(tags)
np.save("./rmrb/train_y", train_y)
np.save("./rmrb/val_y", val_y)

load_bert_repre()
# train_bert, val_bert = load_bert_repre()
# np.savez_compressed(para["data_bert_path"], train_x=train_bert, train_y=train_y, val_x=val_bert, val_y=val_y)