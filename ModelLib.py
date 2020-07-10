import numpy
from keras import Model
from keras.layers import *
from keras_contrib.layers import CRF
from keras_trans_mask import RemoveMask, RestoreMask


def BERT_MODEL(para):
    # for key in para:
    #     print key,para[key]
    bert_input = Input(shape=(para["max_len"], 768,), dtype='float32', name='bert_input')
    # mask = Masking().compute_mask(bert_input)
    mask = Masking()(bert_input)
    repre = Dropout(para["char_dropout"])(mask)
    repre = Dense(200, activation="relu")(repre)
    repre = Bidirectional(LSTM(para["lstm_unit"], return_sequences=True, dropout=para["rnn_dropout"]))(repre)
    removed_mask = RemoveMask()(repre)
    nndata = Conv1D(128, 3, padding='same', strides=1, activation='relu')(removed_mask)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    restored_mask = RestoreMask()([nndata, repre])
    # restored_mask = Dropout(para["cnn_dropout"])(restored_mask)
    crf = CRF(para["tag_num"], sparse_target=True)
    crf_output = crf(restored_mask)
    model = Model(input=bert_input, output=crf_output)
    model.summary()
    # adam_0 = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile("adam", loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def BERT_MODEL_1(para):
    # for key in para:
    #     print key,para[key]
    bert_input = Input(shape=(para["max_len"], 768,), dtype='float32', name='bert_input')
    # mask = Masking().compute_mask(bert_input)
    mask = Masking()(bert_input)
    repre = Dropout(para["char_dropout"])(mask)
    repre = Dense(200, activation="relu")(repre)
    removed_mask = RemoveMask()(repre)
    nndata = Conv1D(128, 3, padding='same', strides=1, activation='relu')(removed_mask)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu')(nndata)
    nndata = Conv1D(64, 3, padding='same', strides=1, activation='relu', dilation_rate=2)(nndata)
    restored_mask = RestoreMask()([nndata, repre])
    repre = Bidirectional(LSTM(para["lstm_unit"], return_sequences=True, dropout=para["rnn_dropout"]))(restored_mask)
    crf = CRF(para["tag_num"], sparse_target=True)
    crf_output = crf(repre)
    model = Model(input=bert_input, output=crf_output)
    model.summary()
    # adam_0 = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile("adam", loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def BERT_MODEL_bak(para):
    # for key in para:
    #     print key,para[key]
    bert_input = Input(shape=(para["max_len"], 768,), dtype='float32', name='bert_input')
    mask = Masking()(bert_input)
    repre = Dropout(para["char_dropout"])(mask)
    repre = Dense(300, activation="relu")(repre)
    repre = Bidirectional(LSTM(para["lstm_unit"], return_sequences=True, dropout=para["rnn_dropout"]))(repre)
    crf = CRF(para["tag_num"], sparse_target=True)
    crf_output = crf(repre)
    model = Model(input=bert_input, output=crf_output)
    model.summary()
    # adam_0 = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile("adam", loss=crf.loss_function, metrics=[crf.accuracy])
    return model


if __name__ == "__main__":
    data = numpy.ones((10, 10, 10), dtype='float32')
    # data_2 = numpy.ones((10, 100),dtype='float32')
    #
    # data_input = Input(shape=(100,))
    # data_reshape = Reshape(target_shape=(10, 10))(data_input)
    # weight_input = Input(shape=(10, 10))
    # # model.add(Embedding(input_dim=3,output_dim=5,weights=[weight],mask_zero=True))
    # output = Multiply(output_dim=(10,10))([data_reshape, weight_input])
    #
    # model = Model(input=[data_input,weight_input], output=output)
    # result = model.predict([data_2, data], batch_size=2)
    # model.summary()

    # train_x, train_y, val_x, val_y, word2id, tags, img_voc = pickle.load(open(config.data_pk, 'rb'))
    #
    # img_embed = load_img_embed(word2id)
    # x = numpy.array([1.0, 2.0, 3.0])
    # # x = x.astype(numpy.int64)
    # # result = tf.nn.embedding_lookup(img_embed, x)
    # # sess = tf.Session()
    # # result = sess.run(result)
    # # print(result)
    #
    # x_input = Input(shape=(train_x[0].shape[1],), dtype="int64")
    # y = MyLayer.ImageEmbeding(img_weight=img_embed, output_dim=(50, 50, 1))(x_input)
    # model = Model(input=x_input, output=y)
    # model.summary()
    # result = model.predict(train_x[0], batch_size=64)
    # print(result.shape)
