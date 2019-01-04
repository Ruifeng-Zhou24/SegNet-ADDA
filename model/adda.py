import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'util'))

from segnet import SegNet
from utils import adversarial_loss


class ADDA(SegNet):

    def __init__(self, args):
        SegNet.__init__(self, args)

        self.s_source = 's'
        self.s_target = 't'
        self.s_discriminator = 'd'
        self.s_generator = 'g'

        self.src_image_path = 'train.txt'
        self.tar_image_path = 'validation400.txt'
        self.ckpt_dir = args.log_dir + args.ckpt
        self.src_ckpt_path = self.ckpt_dir + 'src_model.ckpt'
        self.tar_ckpt_path = self.ckpt_dir + 'tar_model.ckpt'

        self.lr_g = 1e-3
        self.lr_d = 1e-3

        self.generate_models()

    def generate_models(self):
        vars = []
        with tf.Session(config=self.sess_config) as sess:
            tar_vars = []
            src_vars = []
            for name, shape in tf.contrib.framework.list_variables(self.ckpt_dir):
                var = tf.contrib.framework.load_variable(self.ckpt_dir, name)
                vars.append(tf.Variable(var, name=name))
                tar_vars.append(tf.Variable(var, name='t/' + name))
                src_vars.append(tf.Variable(var, name='s/' + name))

            src_saver = tf.train.Saver(src_vars)
            tar_saver = tf.train.Saver(tar_vars)
            tmp_saver = tf.train.Saver(vars)

            sess.run(tf.global_variables_initializer())

            src_saver.save(sess, self.src_ckpt_path)
            print('model saved to {}'.format(self.src_ckpt_path))
            tar_saver.save(sess, self.tar_ckpt_path)
            print('model saved to {}'.format(self.tar_ckpt_path))
            tmp_saver.save(sess, self.ckpt_dir + 'model.ckpt')  # rewrite the checkpoint

        tf.reset_default_graph()  # to avoid variables naming violence

    def discriminator(self, inputs, trainable=True):
        flat = tf.layers.Flatten()(inputs)

        fc1 = tf.layers.dense(flat, 576, activation=tf.nn.leaky_relu, trainable=trainable, name='fc1')
        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.leaky_relu, trainable=trainable, name='fc2')
        fc3 = tf.layers.dense(fc2, 1, activation=None, trainable=trainable, name='fc3')

        return fc3

    def train(self):
        src_image_filenames, src_label_filenames = self.dataset.get_filename_list(self.src_image_path)
        tar_image_filenames, tar_label_filenames = self.dataset.get_filename_list(self.tar_image_path)

        src_image, src_label = self.dataset.batch(self.batch_size, src_image_filenames, src_label_filenames)
        tar_image, tar_label = self.dataset.batch(self.batch_size, tar_image_filenames, tar_label_filenames)

        # for source domain
        with tf.variable_scope(self.s_source):
            src_encode_output = self.encoder(src_image, tf.constant(False))
            src_decode_output = self.decoder(src_encode_output, tf.constant(False))
            src_cls_loss, src_logits = self.classifier(src_decode_output, src_label)
        with tf.variable_scope(self.s_discriminator):
            dis_src = self.discriminator(src_encode_output[0])

        # for target domain
        with tf.variable_scope(self.s_target):
            tar_encode_output = self.encoder(tar_image, phase_train=tf.constant(True))
            tar_decode_output = self.decoder(tar_encode_output, tf.constant(False))
            tar_cls_loss, tar_logits = self.classifier(tar_decode_output, tar_label)
        with tf.variable_scope(self.s_discriminator, reuse=True):
            dis_tar = self.discriminator(tar_encode_output[0])

        # build loss
        g_loss, d_loss = adversarial_loss(dis_src, dis_tar)

        # create optimizer for two task
        var_tar = tf.trainable_variables(self.s_target)
        optim_g = tf.train.AdamOptimizer(self.lr_g).minimize(g_loss, var_list=var_tar)

        var_d = tf.trainable_variables(self.s_discriminator)
        optim_d = tf.train.AdamOptimizer(self.lr_d).minimize(d_loss, var_list=var_d)

        with tf.Session(config=self.sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.s_source)
            self.tar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.s_target)

            src_saver = tf.train.Saver(self.src_vars)
            tar_saver = tf.train.Saver(self.tar_vars)

            src_saver.restore(sess, self.src_ckpt_path)
            print("src model restored successfully!")
            tar_saver.restore(sess, self.tar_ckpt_path)
            print("tar model restored succesfully!")

            for i in range(self.max_steps):
                _, d_loss_, = sess.run([optim_d, d_loss])
                _, g_loss_ = sess.run([optim_g, g_loss])

                if i % 10 == 0:
                    print("step:{}, g_loss:{:.4f}, d_loss:{:.4f}".format(i, g_loss_, d_loss_))
                if i % 100 == 0:
                    print("testing ...")
                    pred = tf.argmax(tar_logits, axis=3)
                    print(pred)
                    _, pred_image = sess.run([src_logits, pred])
                    pred_image = pred_image[0]
                    pred_image[pred_image == 0] = 255
                    pred_image[pred_image == 1] = 128
                    pred_image[pred_image == 2] = 0
                    pred_image = Image.fromarray(np.uint8(pred_image))
                    save_path = self.log_dir + '{}.bmp'.format(i)
                    pred_image.save(save_path)
                    print("image saved to {}".format(save_path))
