# BERT-BiLSTM-IDCNN-CRF
BERT-BiLSTM-IDCNN-CRF的Keras版实现

## BERT配置

1. 首先需要下载Pre-trained的BERT模型

本文用的是Google开源的中文BERT模型：

- https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

2. 安装BERT-as-service

`pip install bert-serving-server bert-serving-client`

源项目如下：

- https://github.com/hanxiao/bert-as-service

3. 启动bert-serving-server
 
打开服务器，在BERT文件夹（chinese_L-12_H-768_A-12）的根目录下，打开终端，输入命令：
 
`bert-serving-start -pooling_strategy NONE -max_seq_len 130 -mask_cls_sep -model_dir chinese_L-12_H-768_A-12/  -num_worker 1`

## 文件描述

- preprocess.py 数据预处理
- train.py 训练模型
- test.py 评估模型
- Modellib.py 模型
- config.py 参数配置

## 模型使用

1. 配置BERT
2. 构建数据集：
    - rmrb目录下为人民日报数据集
    - 分别运行rmrb目录下MyTest1，2，3
    - 原始demo数据集为data目录下，2015词性标注数据集
3. 执行preprocess.py
4. 执行SaveBert.py
5. 训练：执行train.py
6. 评估：执行test.py

# todo

1. 数据处理存在问题

- 机器性能受限，所以bert的数据是先存下来再载入，存的时候也分开存了，关键函数`load_bert_repre`。
- 机器性能受限，句子长度取大于44小于128的，其他的都直接丢弃了，关键函数`_parse_data`。

2. 模型IDCNN部分没有加LayerNormalization

3. 模型无法保证这种结构和超参数下可以达到最佳效果，需要更多测试

# 依赖

python >= 3.5
keras = 2.2.4
tensorflow = 1.14.0
keras-contrib = 2.0.8
bert-as-service

# 致谢及参考

https://github.com/AidenHuen/BERT-BiLSTM-CRF

https://pypi.org/project/keras-trans-mask/


