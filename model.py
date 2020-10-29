# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random

from layers import *

img_layer = 1


def build_resnet_block(inputres, dim, change_dimension=False, block_stride=2, name="resnet"):
    with tf.variable_scope(name):
        if change_dimension:
            short_cut_conv = general_conv3d(inputres, dim, 1, 1, 1, block_stride, block_stride, block_stride, 0.02, "VALID", "sc", do_relu=False)
        else:
            short_cut_conv = inputres
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        if change_dimension:
            out_res = general_conv3d(out_res, dim, 3, 3, 3, block_stride, block_stride, block_stride, 0.02, "VALID", "c1")
        else:
            out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c2", do_relu=False)
        return tf.nn.relu(out_res + short_cut_conv)



def build_generator(inputgen, dim, numofres = 6, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv3d(pad_input, dim, f, f, f, 1, 1, 1, 0.02, name="c1")
        o_c2 = general_conv3d(o_c1, dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv3d(o_c2, dim * 4, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c3")
        o_rb = o_c3
        for idd in range(numofres):
            o_rb = build_resnet_block(o_rb, dim * 4, name='r{0}'.format(idd))
        o_c4 = general_conv3d(o_rb, dim*4, ks, ks, ks, 1, 1, 1, 0.02, "SAME", name="c4")
        o_c5 = general_deconv3d(o_c4, [1, 64, 64, 64, dim * 2], dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c5")
        o_c6 = general_deconv3d(o_c5, [1, 128, 128, 128, dim], dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c6")
        o_c6_pad = tf.pad(o_c6, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c7 = general_conv3d(o_c6_pad, img_layer, f, f, f, 1, 1, 1, 0.02, "VALID", "c7", do_relu=False)
        out_gen = tf.nn.tanh(o_c7, "t1")
        return out_gen


def build_gen_discriminator(inputdisc, dim, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        o_c1 = general_conv3d(inputdisc, dim, f, f, f, 2, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv3d(o_c1, dim * 2, f, f, f, 2, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv3d(o_c2, dim * 4, f, f, f, 2, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv3d(o_c3, dim * 8, f, f, f, 1, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv3d(o_c4, 1, f, f, f, 1, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)
        return o_c5


def build_classifier(input, dim, order=(1, 2), name="cls"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        ks = 3
        o_c0 = general_conv3d(input, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c0")
        o_p0 = tf.nn.max_pool3d(o_c0, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c1 = general_conv3d(o_p0, dim*2, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c1")
        o_p1 = tf.nn.max_pool3d(o_c1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c2 = general_conv3d(o_p1, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c2")
        o_p2 = tf.nn.max_pool3d(o_c2, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c3 = general_conv3d(o_p2, dim*4, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="c3")
        o_p3 = tf.nn.max_pool3d(o_c3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        o_c4 = general_conv3d(o_p3, dim*4, 3, 3, 3, 1, 1, 1, 0.2, "SAME", name="c4")
        o_p4 = tf.nn.avg_pool3d(o_c4, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='VALID')

        if 0 in order:
            feats = tf.reduce_mean(o_p4, axis=(1, 2, 3))
        else:
            descs = []
            if 1 in order:
                descs.append(o_p4)
            if 2 in order:
                descs.append(tf.square(o_p4) - 1)
            if 3 in order:
                descs.append(tf.square(o_p4) * o_p4)
            descs = tf.nn.l2_normalize(tf.concat(descs, axis=-1), axis=-1)
            feats = tf.nn.l2_normalize(
                tf.reshape(descs, (-1, descs.shape[1] * descs.shape[2] * descs.shape[3] * descs.shape[4])), axis=-1)
        logits, prob = fc_op(feats, "fc_layer", 2, activation=tf.nn.softmax)
        return logits, prob
		
def build_generator_classifier(inputgen, dim, numofres = 6):
    with tf.variable_scope("generator"):
        f = 7
        ks = 3
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv3d(pad_input, dim, f, f, f, 1, 1, 1, 0.02, name="c1")
        o_c2 = general_conv3d(o_c1, dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv3d(o_c2, dim * 4, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c3")
        o_rb = o_c3
        for idd in range(numofres):
            o_rb = build_resnet_block(o_rb, dim * 4, name='r{0}'.format(idd))
        o_c4 = general_deconv3d(o_rb, [1, 64, 64, 64, dim * 2], dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv3d(o_c4, [1, 128, 128, 128, dim], dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c5")
        o_c5_pad = tf.pad(o_c5, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv3d(o_c5_pad, img_layer, f, f, f, 1, 1, 1, 0.02, "VALID", "c6", do_relu=False)
        out_gen = tf.nn.tanh(o_c6, "t1")

    with tf.variable_scope("classifier"):
        f1 = tf.concat((o_c1,o_c5),axis=-1)
        f1 = general_conv3d(f1, dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "feature1")
        f2 = tf.concat((o_c2,o_c4),axis=-1)
        f2 = general_conv3d(f2, dim, ks, ks, ks, 1, 1, 1, 0.02, "SAME", "feature2")
        f2 = tf.concat((f1,f2),axis=-1)
        f2 = tf.nn.max_pool3d(f2, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        f3 = general_conv3d(f2, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="feature3")
        f3 = tf.nn.max_pool3d(f3, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='SAME')
        f4 = general_conv3d(f3, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="feature4")
        f4 = tf.nn.max_pool3d(f4, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='VALID')
        f5 = general_conv3d(f4, dim, ks, ks, ks, 1, 1, 1, 0.2, "SAME", name="feature5")
        f5 = tf.nn.avg_pool3d(f5, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], padding='VALID')
        descs = tf.nn.l2_normalize(tf.concat((f5, tf.square(f5) - 1), axis=-1), axis=-1)
        feats = tf.nn.l2_normalize(tf.reshape(descs, (-1, descs.shape[1]*descs.shape[2]*descs.shape[3]*descs.shape[4])), axis=-1)
        logits, prob = fc_op(feats, "fc_layer", 2, activation=tf.nn.softmax)
    return out_gen, logits, prob

