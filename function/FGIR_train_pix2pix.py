# --------------------------------------------------------
# MXNet Implementation of pix2pix GAN
# Copyright (c) 2017 UIUC
# Written by Bowen Cheng
# --------------------------------------------------------

import cv2
import time
import argparse
import logging
import pprint
import os
import sys
import matplotlib
matplotlib.use('Agg')
import time
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train PIX2PIX')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../../external/mxnet', config.MXNET_VERSION))

import shutil
import numpy as np
import mxnet as mx

from symbols.pix2pix_instance import defineG_encoder_decoder, defineG_unet, defineD_n_layers, defineD_basic
from symbols.pix2pix_batch import defineG_encoder_decoder_batch, defineG_unet_batch, defineD_n_layers_batch, defineD_basic_batch
from core.create_logger import create_logger
#from core.loader import pix2pixIter
from data_loader.GAN_data_loader import DataIter
from core.visualize import visualize
from core import metric
from core.lr_scheduler import PIX2PIXScheduler


def main():
    # set debug
    DEBUG = False
    # =============setting============
    dataset = config.dataset.dataset
    batch_size = config.TRAIN.BATCH_SIZE
    lr = config.TRAIN.lr
    beta1 = config.TRAIN.beta1
    sigma = 0.02
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    assert len(ctx) == 1, 'Multi GPU not supported.'
    ctx = ctx[0]
    frequent = config.default.frequent
    check_point = True

    logger, final_output_path = create_logger(config.output_path, args.cfg)
    prefix = os.path.join(final_output_path, config.TRAIN.model_prefix)
    train_fig_path = os.path.join(final_output_path, 'train_fig')

    train_fig_prefix = os.path.join(train_fig_path, dataset)

    if not os.path.exists(train_fig_path):
        os.makedirs(train_fig_path)

    # set random seed for reproducibility
    mx.random.seed(config.RNG_SEED)
    np.random.seed(config.RNG_SEED)

    # ==============data==============
    #train_data = pix2pixIter(config, shuffle=True, ctx=ctx)
    train_data = DataIter(config,ctx=ctx)

    step = config.TRAIN.step_epoch * train_data.size / batch_size
    step_decay = config.TRAIN.decay_epoch * train_data.size / batch_size
    if config.TRAIN.end_epoch == (config.TRAIN.step_epoch + config.TRAIN.decay_epoch):
        lr_scheduler_g = PIX2PIXScheduler(step=int(step), step_decay=int(step_decay), base_lr=lr)
        lr_scheduler_d = PIX2PIXScheduler(step=int(step), step_decay=int(step_decay), base_lr=lr/2.0)
    else:
        lr_scheduler_g = None
        lr_scheduler_d = None

    label = mx.nd.zeros((batch_size,), ctx=ctx)

    # print config
    pprint.pprint(config)
    logger.info('system:{}'.format(os.uname()))
    logger.info('mxnet path:{}'.format(mx.__file__))
    logger.info('rng seed:{}'.format(config.RNG_SEED))
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # =============Generator Module=============
    if batch_size == 1:
        if config.netG == 'autoencoder':
            generatorSymbol = defineG_encoder_decoder(config)
        elif config.netG == 'unet':
            generatorSymbol = defineG_unet(config)
        else:
            raise NotImplemented
    else:
        if config.netG == 'autoencoder':
            generatorSymbol = defineG_encoder_decoder_batch(config)
        elif config.netG == 'unet':
            generatorSymbol = defineG_unet_batch(config)
        else:
            raise NotImplemented

    if DEBUG:
        generatorGroup = generatorSymbol.get_internals()
        name_list = generatorGroup.list_outputs()
        out_name = []
        for name in name_list:
            if 'output' in name:
                out_name += [generatorGroup[name]]
        out_group = mx.sym.Group(out_name)
        out_shapes = out_group.infer_shape(A=(4, 3, 256, 256))

    generator = mx.mod.Module(symbol=generatorSymbol, data_names=('A', 'B',), label_names=None, context=ctx)
    generator.bind(data_shapes=train_data.provide_data)
    #draw network
    #network_test(generatorSymbol)
    # init params
    arg_params = {}
    aux_params = {}
    arg_names = generatorSymbol.list_arguments()
    aux_names = generatorSymbol.list_auxiliary_states()
    arg_shapes, _, aux_shapes = generatorSymbol.infer_shape(A = train_data.provide_data[0][1],
                                                            B = train_data.provide_data[1][1])

    if batch_size == 1:
        for idx, arg_name in enumerate(arg_names):
            if 'weight' in arg_name:
                arg_params[arg_name] = mx.random.normal(0.0, sigma, shape=arg_shapes[idx])
            elif 'gamma' in arg_name:
                arg_params[arg_name] = mx.random.normal(1.0, sigma, shape=arg_shapes[idx])
            elif 'bias' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            elif 'beta' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            else:
                # raise NameError('Unknown parameter name.')
                pass
    else:
        for idx, arg_name in enumerate(arg_names):
            if 'weight' in arg_name:
                arg_params[arg_name] = mx.random.normal(0.0, sigma, shape=arg_shapes[idx])
            elif 'gamma' in arg_name:
                arg_params[arg_name] = mx.random.normal(1.0, sigma, shape=arg_shapes[idx])
            elif 'bias' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            elif 'beta' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            else:
                # raise NameError('Unknown parameter name.')
                pass
        for idx, aux_name in enumerate(aux_names):
            if 'mean' in aux_name:
                aux_params[aux_name] = mx.nd.zeros(shape=aux_shapes[idx])
            elif 'var' in aux_name:
                aux_params[aux_name] = mx.nd.ones(shape=aux_shapes[idx])
            else:
                raise NameError('Unknown aux_name.')

    generator.init_params(arg_params=arg_params, aux_params=aux_params)

    if lr_scheduler_g is not None:
        generator.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': lr,
                'lr_scheduler': lr_scheduler_g,
                'beta1': beta1,
                'rescale_grad': 1.0/batch_size
            })
    else:
        generator.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': lr,
                'beta1': beta1,
                'rescale_grad': 1.0/batch_size
            })
    mods = [generator]

    # =============Discriminator Module=============
    if batch_size == 1:
        if config.netD == 'basic':
            discriminatorSymbol = defineD_basic()
        elif config.netD == 'n_layers':
            discriminatorSymbol = defineD_n_layers(n_layers = config.n_layers)
        else:
            raise NotImplemented
    else:
        if config.netD == 'basic':
            discriminatorSymbol = defineD_basic_batch(batch_size=batch_size)
        elif config.netD == 'n_layers':
            discriminatorSymbol = defineD_n_layers_batch(n_layers = config.n_layers, batch_size=batch_size)
        else:
            raise NotImplemented

    if DEBUG:
        generatorGroup = discriminatorSymbol.get_internals()
        name_list = generatorGroup.list_outputs()
        out_name = []
        for name in name_list:
            if 'output' in name:
                out_name += [generatorGroup[name]]
        out_group = mx.sym.Group(out_name)
        out_shapes = out_group.infer_shape(A=(1, 3, 256, 256), B=(1, 3, 256, 256))

    discriminator = mx.mod.Module(symbol=discriminatorSymbol, data_names=('A', 'B',), label_names=('label',), context=ctx)
    discriminator.bind(data_shapes=train_data.provide_data,
                       label_shapes=[('label', (batch_size,))],
                       inputs_need_grad=True)

    # init params
    arg_params = {}
    aux_params = {}
    arg_names = discriminatorSymbol.list_arguments()
    aux_names = discriminatorSymbol.list_auxiliary_states()
    arg_shapes, _, aux_shapes = discriminatorSymbol.infer_shape(A=train_data.provide_data[0][1],
                                                                B=train_data.provide_data[1][1],
                                                                label=(batch_size,))

    if batch_size == 1:
        for idx, arg_name in enumerate(arg_names):
            if 'weight' in arg_name:
                arg_params[arg_name] = mx.random.normal(0.0, sigma, shape=arg_shapes[idx])
            elif 'gamma' in arg_name:
                arg_params[arg_name] = mx.random.normal(1.0, sigma, shape=arg_shapes[idx])
            elif 'bias' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            elif 'beta' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            else:
                # raise NameError('Unknown parameter name.')
                pass
    else:
        for idx, arg_name in enumerate(arg_names):
            if 'weight' in arg_name:
                arg_params[arg_name] = mx.random.normal(0.0, sigma, shape=arg_shapes[idx])
            elif 'gamma' in arg_name:
                arg_params[arg_name] = mx.random.normal(1.0, sigma, shape=arg_shapes[idx])
            elif 'bias' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            elif 'beta' in arg_name:
                arg_params[arg_name] = mx.nd.zeros(shape=arg_shapes[idx])
            else:
                # raise NameError('Unknown parameter name.')
                pass
        for idx, aux_name in enumerate(aux_names):
            if 'mean' in aux_name:
                aux_params[aux_name] = mx.nd.zeros(shape=aux_shapes[idx])
            elif 'var' in aux_name:
                aux_params[aux_name] = mx.nd.ones(shape=aux_shapes[idx])
            else:
                raise NameError('Unknown aux_name.')

    discriminator.init_params(arg_params=arg_params, aux_params=aux_params)

    # gradient is scaled in LogisticRegression layer, no need to rescale gradient
    if lr_scheduler_d is not None:
        discriminator.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': lr / 2.0,
                'lr_scheduler': lr_scheduler_d,
                'beta1': beta1,
                'rescale_grad': 1.0
            })
    else:
        discriminator.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': lr / 2.0,
                'beta1': beta1,
                'rescale_grad': 1.0
            })
    mods.append(discriminator)

    # metric
    mG = metric.CrossEntropyMetric()
    mD = metric.CrossEntropyMetric()
    mACC = metric.AccMetric()
    mL1 = metric.L1LossMetric(config)

    t_accumulate = 0

    # =============train===============
    for epoch in range(config.TRAIN.end_epoch):
        train_data.reset()
        mACC.reset()
        mG.reset()
        mD.reset()
        mL1.reset()
        for t, batch in enumerate(train_data):

            t_start = time.time()

            # generator input real A, output fake B
            generator.forward(batch, is_train=True)
            outG = generator.get_outputs()

            # update discriminator on fake
            # discriminator input real A and fake B
            # want discriminator to predict fake (0)
            label[:] = 0
            discriminator.forward(mx.io.DataBatch([batch.data[0], outG[1]], [label]), is_train=True)
            discriminator.backward()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in discriminator._exec_group.grad_arrays]

            discriminator.update_metric(mD, [label])
            discriminator.update_metric(mACC, [label])

            # update discriminator on real
            # discriminator input real A and real B
            # want discriminator to predict real (1)
            label[:] = 1
            batch.label = [label]
            discriminator.forward(batch, is_train=True)
            discriminator.backward()
            for gradsr, gradsf in zip(discriminator._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    # gradr =  (gradr + gradf)/2
                    gradr += gradf
            discriminator.update()

            discriminator.update_metric(mD, [label])
            discriminator.update_metric(mACC, [label])

            # update generator
            # discriminator input real A and fake B
            # want discriminator to predict real (1)
            label[:] = 1
            discriminator.forward(mx.io.DataBatch([batch.data[0], outG[1]], [label]), is_train=True)
            discriminator.backward()
            diffD = discriminator.get_input_grads()
            # loss does not need output gradient
            generator.backward([mx.nd.array(np.ones((batch_size,)), ctx=ctx), diffD[1] * config.GAN_loss])
            generator.update()

            mG.update([label], discriminator.get_outputs())
            mL1.update(None, outG)

            t_accumulate += time.time() - t_start

            t += 1
            if t % frequent == 0:
                if config.TRAIN.batch_end_plot_figure:
                    visualize(batch.data[0].asnumpy(), batch.data[1].asnumpy(), outG[1].asnumpy(), train_fig_prefix + '-train-%04d-%06d.png' % (epoch + 1, t))
                print 'Epoch[{}] Batch[{}] Time[{:.4f}] dACC: {:.4f} gCE: {:.4f} dCE: {:.4f} gL1: {:.4f}'.format(epoch, t, t_accumulate, mACC.get()[1], mG.get()[1], mD.get()[1], mL1.get()[1])
                logger.info('Epoch[{}] Batch[{}] Speed[{:.4f} batch/s] dACC: {:.4f} gCE: {:.4f} dCE: {:.4f} gL1: {:.4f}\n'.format(epoch, t, frequent * batch_size / t_accumulate, mACC.get()[1], mG.get()[1], mD.get()[1], mL1.get()[1]))
                t_accumulate = 0

        if check_point:
            print('Saving...')
            if config.TRAIN.epoch_end_plot_figure:
                visualize(batch.data[0].asnumpy(), batch.data[1].asnumpy(), outG[1].asnumpy(),
                          train_fig_prefix + '-train-%04d.png' % (epoch + 1))
            if (epoch + 1) % config.TRAIN.save_interval == 0:
                generator.save_params(prefix + '-generator-%04d.params' % (epoch + 1))
                discriminator.save_params(prefix + '-discriminator-%04d.params' % (epoch + 1))

    generator.save_params(prefix + '-generator-%04d.params' % config.TRAIN.end_epoch)
    discriminator.save_params(prefix + '-discriminator-%04d.params' % config.TRAIN.end_epoch)

if __name__ == '__main__':
    main()