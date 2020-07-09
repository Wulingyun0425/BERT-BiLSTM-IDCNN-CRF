# -*- coding: UTF-8 -*-

from __future__ import print_function
import codecs
import re


def sentence2split():
    with open('./renmin3.txt', 'r') as inp, codecs.open('./renmin4.txt', 'w') as outp:
        line_list = list(inp)
        new_list = []
        length = len(line_list)
        for index in range(length):
            new_list.append(line_list[index])
            if index + 1 < length and line_list[index][0] in ["。", "!", "！", "?", "？"] and line_list[index+1] != "\n":
                new_list.append("\n")
        # new_list.append("\n")
        outp.write("".join(new_list))
        print("done")


def splitdataset():
    with open('./renmin4.txt', 'r') as inp, codecs.open('./my_train.txt', 'w') as out_train, codecs.open(
            './my_dev.txt', 'w') as out_dev, codecs.open('./my_test.txt', 'w') as out_test:
        cotent = inp.read()
        line_list = cotent.split("\n\n")
        new_list = []
        for line in line_list:
            if len(line) > 3*50:
                new_list.append(line)
        length = len(new_list)
        train_flag = int(length * 0.8)
        dev_flag = int(length * 0.9)
        # train_list = line_list[:train_flag]
        train_list = new_list[:dev_flag]
        dev_list = new_list[train_flag:dev_flag]
        test_list = new_list[dev_flag:]
        out_train.write("\n\n".join(train_list).replace("\n\n\n", "\n\n"))
        out_dev.write("\n\n".join(dev_list).replace("\n\n\n", "\n\n"))
        out_test.write("\n\n".join(test_list).replace("\n\n\n", "\n\n"))
        print("done")


sentence2split()
splitdataset()
