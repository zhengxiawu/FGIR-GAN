# --------------------------------------------------------
# MXNet Implementation of pix2pix GAN
# Copyright (c) 2017 UIUC
# Written by Bowen Cheng
# --------------------------------------------------------

import mxnet as mx
import os
import numpy as np
import cv2
import random
from data_basic import *
#read txt function
def get_data_by_txt(txt_path):
    data_gen = []
    label = []
    with open(txt_path) as f:
        txt_lines = f.readlines()
    for key,value in enumerate(txt_lines):
        info_list = value.split(' ')
        data_gen.append(info_list[0])
        label.append(int(info_list[1]))
    return data_gen,label
def get_label_idx(label):
    unique_label = list(set(label))
    label_idx = []
    for i in unique_label:
        label_idx.append([j for j,x in enumerate(label) if x == i])
    return unique_label,label_idx

class DataIter(mx.io.DataIter):
    def __init__(self, config, shuffle=False, ctx=None, is_train=True):
        self.is_train = is_train
        self.config = config
        self.dataset = config.dataset.dataset  # name of dataset
        self.imageset = config.dataset.imageset  # name of image name text file
        self.testset = config.dataset.testset
        self.root = os.path.join(config.dataset.root, config.dataset.dataset)  # path to store image name text file
        self.image_root = os.path.join(config.dataset.image_root, config.dataset.dataset)  # path to jpeg file
        self.image_files = self._load_image_path()
        self.data_gen, self.label = get_data_by_txt(self.txt_path)
        self.unique_label, self.unique_label_idx = get_label_idx(self.label)
        self.img_mean = np.array([self.config.dataset.mean_r,
                                  self.config.dataset.mean_g,
                                  self.config.dataset.mean_b])
        self.size = len(self.image_files)
        self.index = np.arange(self.size)

        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]

        self.cur = 0
        self.shuffle = shuffle

        self.batch_size = config.TRAIN.BATCH_SIZE
        # assert self.batch_size == 1

        self.AtoB = config.AtoB
        self.A = None
        self.B = None
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        if self.is_train:
            return [('A', (self.batch_size, 3, self.config.fineSize, self.config.fineSize)),
                    ('B', (self.batch_size, 3, self.config.fineSize, self.config.fineSize))]
        else:
            return [('A', (self.batch_size, 3, self.config.TEST.img_h, self.config.TEST.img_w)),
                    ('B', (self.batch_size, 3, self.config.TEST.img_h, self.config.TEST.img_w))]

    @property
    def provide_label(self):
        return [('label',(self.batch_size * 2,))]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return Batch(data=[mx.nd.array(self.A, ctx=self.ctx),
                               mx.nd.array(self.B, ctx=self.ctx)],
                         label=[mx.nd.array(self.label_batch)])
            # return mx.io.DataBatch(data=[mx.nd.array(self.A, ctx=self.ctx),
            #                              mx.nd.array(self.B, ctx=self.ctx)],
            #                        label=[mx.nd.array(self.label_batch)],
            #                        pad=self.getpad(), index=self.getindex(),
            #                        provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur > self.size:
            return self.cur - self.size
        else:
            return 0

    def _load_image_path(self):
        if self.is_train:
            fname = os.path.join(self.root, self.imageset + '.txt')
        else:
            fname = os.path.join(self.root, self.testset + '.txt')
        assert os.path.exists(fname), 'Path does not exist: {}'.format(fname)
        self.txt_path = fname
        with open(fname) as f:
            lines = [x.strip() for x in f.readlines()]
        return lines

    def get_batch(self):
        self.label_batch = []
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        pad = cur_from + self.batch_size - cur_to

        if self.is_train:
            batchA = np.zeros((0, 3, self.config.fineSize, self.config.fineSize))
            batchB = np.zeros((0, 3, self.config.fineSize, self.config.fineSize))
        else:
            batchA = np.zeros((0, 3, self.config.TEST.img_h, self.config.TEST.img_w))
            batchB = np.zeros((0, 3, self.config.TEST.img_h, self.config.TEST.img_w))

        for index in range(cur_from, cur_to):
            #random sample label
            label_rand_idx = random.sample(range(len(self.unique_label)), 1)[0]
            data_dix = random.sample(self.unique_label_idx[label_rand_idx], 2)
            assert  self.label[data_dix[0]] == self.label[data_dix[1]]
            self.label_batch.append(self.label[data_dix[0]])
            self.label_batch.append(self.label[data_dix[1]])
            A = cv2.imread(self.data_gen[data_dix[0]])
            B = cv2.imread(self.data_gen[data_dix[1]])
            A = cv2.resize(A,(self.config.loadSize , self.config.loadSize))
            B = cv2.resize(B, (self.config.loadSize, self.config.loadSize))
            A = A.astype('float32')
            B = B.astype('float32')
            A = A - self.img_mean
            B = B - self.img_mean
            A = np.transpose(A[..., np.newaxis], (3, 2, 0, 1))
            B = np.transpose(B[..., np.newaxis], (3, 2, 0, 1))
            batchA = np.concatenate((batchA, A), axis=0)
            batchB = np.concatenate((batchB, B), axis=0)
            self.A = A
            self.B = B

        if pad > 0:
            if self.is_train:
                self.A = np.concatenate((self.A, np.zeros((pad, 3, self.config.fineSize, self.config.fineSize))), axis=0)
                self.B = np.concatenate((self.B, np.zeros((pad, 3, self.config.fineSize, self.config.fineSize))), axis=0)
            else:
                self.A = np.concatenate((self.A, np.zeros((pad, 3, self.config.TEST.img_h, self.config.TEST.img_w))), axis=0)
                self.B = np.concatenate((self.B, np.zeros((pad, 3, self.config.TEST.img_h, self.config.TEST.img_w))), axis=0)
