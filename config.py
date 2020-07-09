para = {}

# para["train_path"] = "./data/Pos_train.txt"
# para["test_path"] = "./data/Pos_test.txt"
# para["data_pk_path"] = "./data/my.pk"
# para["data_bert_path"] = "./data/bert"
para["train_path"] = "./rmrb/my_train.txt"
para["test_path"] = "./rmrb/my_test.txt"
para["data_pk_path"] = "./rmrb/my_data.pk"
para["data_bert_path"] = "./rmrb/my_bert"
para["model_path"] = "./data/bert-lstm-crf"

para["max_len"] = 128
para["EPOCHS"] = 40
para["batch_size"] = 16
para["sep"] = "\t"

para["embed_dim"] = 200
para["unit_num"] = 200
para["split_seed"] = 2020
para["traditional_chinese"] = False
para["char_dropout"] = 0.4
para["rnn_dropout"] = 0.4
para["cnn_dropout"] = 0.4
para["lstm_unit"] = 300
para["REPRE_NUM"] = 128

para["fea_dropout"] = 0.3
para["fea_lstm_unit"] = 32
para["fea_dim"] = 20
para["radical_max"] = 7
para["pinyin_max"] = 8
para["rad_max"] = 1



