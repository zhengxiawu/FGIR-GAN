# -*- coding: utf-8 -*-
#@Time    :  8:03 PM
#@Author  : Sherwood
'''
data basic class
'''
import mxnet as mx
import os
class Batch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
class DateIter(mx.io.DataIter):
    def __init__(self):
        pass
    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
