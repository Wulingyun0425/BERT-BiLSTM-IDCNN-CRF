# -*- coding: UTF-8 -*-

from __future__ import print_function
import codecs


def noNullLine():
    with open('./renmin.txt', 'r') as inp, codecs.open('./renmin_new.txt', 'w') as outp:
        line_list = list(inp)
        # for index in range(len(line_list)):
        # line_list.remove("\n")
        outp.write("".join(list(filter(lambda x: x != "\n", line_list))))
        print("done")


noNullLine()
