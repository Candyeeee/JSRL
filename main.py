import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from scipy import io as sio
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from sklearn import metrics
import os, re
import shutil
from PIL import Image
import random
import time
import sys
import csv
from layers import *
from model import *
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow  conversion prediction of SCD')
    parser.add_argument('--lr', default=2*1e-4, type=float, help='learning rate')
    parser.add_argument('--gpu_ids', type=str, default='7', help='gpu ids for GPU')
    parser.add_argument('--max_epoch', type=int, default=101, help='the number of epoch')
    parser.add_argument('--max_image', type=int, default=10000, help='the mamunumber of images')
    parser.add_argument('--to_restore', type=str, default=True, help='restore the checkpoint')
    parser.add_argument('--use_mask', type=str, default=False, help='..')
    parser.add_argument('--cycload', type=str, default=True, help='..')
    parser.add_argument('--save_training_images', type=str, default=True, help='save training images')
    parser.add_argument('--model_stats', type=str, default='testMRI', help='model state')
    parser.add_argument('--input_path', type=str, default='.input/Real/', help='mri input path')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_joint', help='checkpoint dir')
    parser.add_argument('--outpath', type=str, default='./outpath_joint/', help='output path')
    parser.add_argument('--grps', type=str, default=['pCN','sCN','MCI','pMCI','sMCI'], help='output path')
    args = parser.parse_args()
    return args


class CCGAN(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.img_width = 144
        self.img_height = 176
        self.img_depth = 144
        self.img_layer = 1
        self.ngf = 16
        self.ndf = 32
        self.argument_side = 3
        self.selected_feat = 0, 1, 2, 3, 4,
        self.max_epoch = args.max_epoch
        self.max_image = args.max_image
        self.to_restore = args.to_restore
        self.use_mask = args.use_mask
        self.cycload = args.cycload
        self.ckpt_dir = args.ckpt_dir
        self.input_path = args.input_path
        self.outpath = args.outpath
        self.model_stats = args.model_stats
        self.grps = args.grps
        self.tasks = 'cls', 'dis',
        self.groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 1, 'sSCD': 0, 'pSCD': 1,
          'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0,
          'sCN': 0, 'pCN': 1, 'ppCN': 1, 'Autism': 1, 'Control': 0}
        self.lr = args.lr

    def inputAB(self, imdb, cycload=True, augment=True):

        flnm, grp = imdb
        if flnm in self.datapool:
            mdata, pdata, label = self.datapool[flnm]
        else:
            label = np.zeros(2, np.float32)
        cls = self.groups[grp]
        if cls in [0, 1]: label[cls] = 1
        mfile = 'MRI/' + flnm + '.mat'
        pfile = 'PET/' + flnm + '.mat'
        if os.path.exists(self.input_path + mfile):
            mdata = np.array(sio.loadmat(self.input_path + mfile)['IMG'])
        else:
            mdata = None
            print(mfile)
        if os.path.exists(self.input_path + pfile):
            pdata = np.array(sio.loadmat(self.input_path + pfile)['IMG'])
        else:
            pdata = None
            print(pfile)
        if cycload:
            self.datapool[flnm] = mdata, pdata, label

        if augment:
            idx = random.randint(-self.argument_side, self.argument_side)
            idy = random.randint(-self.argument_side, self.argument_side)
            idz = random.randint(-self.argument_side, self.argument_side)
        else:
            idx = 0
            idy = 0
            idz = 0

        if mdata is None:
            im_m = None
        else:
            im_m = mdata[np.newaxis, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis]
            im_m = np.minimum(1, im_m.astype(np.float32) / 96 - 1.0)
        if pdata is None:
            im_p = None
        else:
            im_p = pdata[np.newaxis, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis]
            im_p = im_p.astype(np.float32) / 128 - 1.0
        labels = label[np.newaxis, :]
        return im_m, im_p, labels

    def get_database(self, imdbname, vldgrp=("AD", "CN")):
        imdb = []
        with open(imdbname, newline='') as csvfile:
            imdbreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in imdbreader:
                if row[2] in vldgrp and row[2] != "CN" and row[2] != "SMC":
                    imdb.append(row[1:3])
        return imdb

    def input_setup_adni(self):
        self.datapool = {}
        self.imdb_train = self.get_database('./ADNI1_imdb_36m_psCN.csv', self.grps) + self.get_database('./ADNI2_imdb_36m_psCN.csv', self.grps)
        self.imdb_test  = self.get_database('./ADNI2_imdb_36m_psCN.csv', ['AD'])
        self.imdb_val  = self.get_database('./sSCDpSCDwHC.csv', ['pSCD','sSCD'])
        print(len(self.imdb_train))
        print(len(self.imdb_test))
        print(len(self.imdb_val))

    def model_setup(self):
        self.input_A = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.img_depth, self.img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.img_depth, self.img_layer], name="input_B")
        self.label_holder = tf.placeholder(tf.float32, [None, 2], name="label")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)


        with tf.variable_scope("GAN") as scope:
            self.fake_B,self.logit, self.prob = build_generator_classifier(self.input_A, self.ngf, numofres=3)	# 生成的PET
            self.rec_B = build_gen_discriminator(self.input_B, self.ndf, "d")	#判别realPET
            scope.reuse_variables()
            self.fake_rec_B = build_gen_discriminator(self.fake_B, self.ndf, "d")	#判别syntheticPET


    def loss_calc(self):

        self.model_vars = tf.trainable_variables()
        for var in self.model_vars: print(var.name)	#print所有模型参数
        self.cls_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit, labels=self.label_holder))
        optimizer1 = tf.train.GradientDescentOptimizer(0.01)

        if self.use_mask:
            msk_A = tf.cast(self.input_A > - 1, tf.float32)
            msk_B = tf.cast(self.input_B > - 1, tf.float32)
        else:
            msk_A = 1
            msk_B = 1

        p2p_loss = tf.reduce_mean(tf.abs(self.input_B - self.fake_B)*msk_B)

        disc_loss = tf.reduce_mean(tf.abs(self.fake_rec_B - 1))

        self.g_loss1 = p2p_loss + disc_loss
        self.g_loss2 = p2p_loss + disc_loss + self.cls_loss
        self.d_loss = (tf.reduce_mean(tf.abs(self.fake_rec_B)) + tf.reduce_mean(tf.abs(self.rec_B-1))) / 2.0	#cycA&realA
        optimizer2 = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        d_vars = [var for var in self.model_vars if 'd' in var.name]
        g_vars = [var for var in self.model_vars if 'generator' in var.name]
        c_vars = [var for var in self.model_vars if 'classifier' in var.name]
        gc_vars = g_vars + c_vars

        self.d_trainer = optimizer2.minimize(self.d_loss, var_list=d_vars)
        self.g_trainer1 = optimizer2.minimize(self.g_loss1, var_list=g_vars)
        self.c_trainer = optimizer1.minimize(self.cls_loss, var_list=c_vars)
        self.g_trainer2 = optimizer1.minimize(self.g_loss2, var_list=gc_vars)



    def save_training_images(self, sess, epoch):

        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        for ptr in range(0, self.max_images):
            inputA, inputB, label = self.inputAB(self.imdb_train[ptr], cycload=True, augment=False)
            if (inputA is not None) & (inputB is not None):
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                        [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                        feed_dict={self.input_A: inputA[0:1,:,:,:], self.input_B: inputB[0:1,:,:,:]})
                sio.savemat(self.outpath+"/fake_" + str(epoch) + "_" + str(ptr) + ".mat",
                            {'fake_A': fake_A_temp[0], 'fake_B': fake_B_temp[0],
                             'cyc_A': cyc_A_temp[0], 'cyc_B': cyc_B_temp[0],
                             'input_A': inputA[0], 'input_B': inputB[0]})
                break

    def matrics_calc(self, testvals, labels, pos=0, neg=1):
        mean = np.mean(testvals, axis=0)
        print(np.sum(testvals, axis=0))
        AUC = metrics.roc_auc_score(y_score=np.transpose(testvals), y_true=np.transpose(labels), average='samples')
        TP = 0; TN=0; FP=0; FN=0
        f = 1.00
        for idx in range(len(testvals)):
            if (labels[idx][pos]*f > labels[idx][neg]) & (testvals[idx][pos]*f > testvals[idx][neg]):
                TP = TP + 1
            if (labels[idx][pos]*f < labels[idx][neg]) & (testvals[idx][pos]*f <= testvals[idx][neg]):
                TN = TN + 1
            if (labels[idx][pos]*f < labels[idx][neg]) & (testvals[idx][pos]*f > testvals[idx][neg]):
                FP = FP + 1
            if (labels[idx][pos]*f > labels[idx][neg]) & (testvals[idx][pos]*f <= testvals[idx][neg]):
                FN = FN + 1

        print(TP, FN, TN, FP)
        ACC = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        SEN = (TP) / (TP + FN + 1e-6)
        SPE = (TN) / (TN + FP + 1e-6)
        PPV = (TP) / (TP + FP + 1e-6)
        F_score = (2 * SEN * PPV) / (SEN + PPV + 1e-6)
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)+ 1e-6)
        return [AUC, ACC, SEN, SPE, F_score, MCC]

    def train(self):
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #load model
        with tf.Session(config=config) as sess:
            sess.run(init)

            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(self.ckpt_dir,ckpt_name))
                start_epoch = int(next(re.finditer("(\d+)", ckpt_name)).group()) + 1
                print(" [*] Load SUCCESS")
                print(ckpt_name)
            else:
                start_epoch = 0
                print(" [!] Load failed...")

            for epoch in range(start_epoch, self.max_epoch):
                print("In the epoch ", epoch)

                trainlabels = []; trainprobs = []; losses = 0

                for ptr in range(0, min(self.max_image, len(self.imdb_train))):
                    #print("In the iteration ", ptr, self.imdb_train[ptr])
                    inputA, inputB, label = self.inputAB(self.imdb_train[ptr], cycload=True, augment=True)

                    if (inputA is not None) and (inputB is not None) and (epoch < 50):
                        _, g1_loss = sess.run([self.g_trainer1, self.g_loss1],
                                                    feed_dict={self.input_A: inputA, self.input_B: inputB, self.label_holder:label})
                        _, d_loss = sess.run([self.d_trainer, self.d_loss],
                                     feed_dict={self.input_A: inputA, self.input_B: inputB})
                    if (inputA is not None) and (inputB is not None) and epoch>=50:
                        _, logit, prob, loss = sess.run([self.g_trainer2, self.logit, self.prob, self.cls_loss],
                                feed_dict={self.input_A: inputA, self.input_B: inputB, self.label_holder:label})
                        losses = losses + loss; trainlabels.append(label); trainprobs.append(prob)
                        #_, d_loss = sess.run([self.d_trainer,self.d_loss], feed_dict={self.input_A: inputA, self.input_B: inputB})

                if epoch>=50:
                    print('loss:', losses / len(trainprobs), self.matrics_calc(np.concatenate(trainprobs), np.concatenate(trainlabels), pos=1, neg=0))
                saver.save(sess, "%s/%d-model.ckpt" % (self.ckpt_dir, epoch))


    def test(self):
        ''' Testing Function'''
        print("Testing the results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            sess.run(init)
            epoch = 49
            ckpt_fname = self.ckpt_dir + '/%d-model.ckpt' % epoch
            saver.restore(sess, ckpt_fname)
            if not os.path.exists(self.outpath):
                os.makedirs(self.outpath)
            MAE = []; SSIM = []; PSNR = []
            for ptr in range(min(len(self.imdb_test), self.max_image)):
                inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=False, augment=False)
                if inputA is not None and inputB is not None:
                    filename = self.imdb_test[ptr][0]
                    fakeB = sess.run(self.fake_B, feed_dict={self.input_A: inputA})
                    MAE.append(np.mean(np.abs(fakeB-inputB)))
                    SSIM.append(ssim(inputB[0], fakeB[0], multichannel=True))
                    PSNR.append(psnr(inputB[0]/2, fakeB[0]/2))
                    print(filename)
            print(np.mean(MAE, axis=0), np.mean(SSIM, axis=0), np.mean(PSNR, axis=0))
            print(np.std(MAE, axis=0), np.std(SSIM, axis=0), np.std(PSNR, axis=0))




    def eval(self):
        print("eval the classification results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(50, 101, 1):
                # chkpt_fname = tf.train.latest_checkpoint(check_dir)
                ckpt_fname = self.ckpt_dir+'/%d-model.ckpt'%epoch
                print("epoch-{0}".format(epoch), ckpt_fname)
                saver.restore(sess, ckpt_fname)
                testlabels = []; testprobs = []; losses = 0

                for ptr in range(min(len(self.imdb_val), self.max_image)):
                    inputA, inputB, label = self.inputAB(self.imdb_val[ptr], cycload=True, augment=False)
                    print(self.imdb_val[ptr])
                    if inputA is not None:
                        prob, loss = sess.run([self.prob, self.cls_loss], feed_dict={self.input_A: inputA, self.label_holder: label})
                        losses = losses + loss; testlabels.append(label[0:1]); testprobs.append(prob)
                print('loss:', losses / len(testprobs),
                      self.matrics_calc(np.concatenate(testprobs), np.concatenate(testlabels), pos=1, neg=0))


    def eval2(self):
        print("eval the classification results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            best_epoch = 65
            ckpt_fname = self.ckpt_dir + '/%d-model.ckpt' % best_epoch
            print("epoch-{0}".format(best_epoch), ckpt_fname)
            saver.restore(sess, ckpt_fname)
            testlabels = [];
            testprobs = [];
            losses = 0

            for ptr in range(min(len(self.imdb_val), self.max_image)):
                inputA, inputB, label = self.inputAB(self.imdb_val[ptr], cycload=True, augment=False)
                print(self.imdb_val[ptr])
                if inputA is not None:
                    prob, loss = sess.run([self.prob, self.cls_loss],
                                          feed_dict={self.input_A: inputA, self.label_holder: label})
                    losses = losses + loss;
                    testlabels.append(label[0:1]);
                    testprobs.append(prob)
            sio.savemat('./testprobs-joint',{'prob':testprobs,'label':testlabels})
            

def main():
    args = parse_args()
    if args is None:
        exit()
    model_params = vars(args)
    for k, v in model_params.items():
        print("\t%s:%s"%(k,v))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    with tf.Session() as sess:
        model = CCGAN(sess,args)
        model.input_setup_adni()
        model.model_setup()
        model.loss_calc()
        if model.model_stats == 'train':
            model.train()
        elif model.model_stats == 'test':
            model.test()
        elif model.model_stats == 'eval':
            model.eval()
        else:
            model.eval2()

		


if __name__ == '__main__':
    main()
