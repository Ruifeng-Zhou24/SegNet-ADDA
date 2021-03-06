import time
import math
import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'util'))

from utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, \
    print_hist_summery, get_hist, per_class_acc, conv_layer_with_bn, Jaccard_loss, unpool_2d, writeImage
from input import Dataset
from utils import adversarial_loss


class SegNet:
    def __init__(self, args):
        # Constants describing the training process.
        self.moving_average_decay = 0.9999      # The decay to use for the moving average.
        self.num_steps_per_decay = 1000         # Epochs after which learning rate decays.
        self.learning_rate_decay_factor = 0.95  # Learning rate decay factor.
        self.intial_learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_examples_per_epoch_for_val = 100
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 765
        self.val_iter = self.num_examples_per_epoch_for_val / self.batch_size
        self.image_h = args.image_h
        self.image_w = args.image_w
        self.image_c = args.image_c
        self.num_classes = args.num_classes  # cup, disc, other
        self.max_steps = args.max_steps
        self.log_dir = args.log_dir
        self.mode = args.mode
        self.train_path = args.train_path
        self.val_path = args.val_path
        self.test_path = args.test_path
        self.test_ckpt = args.log_dir + args.ckpt
        self.ckpt = args.ckpt
        self.output = args.output
        self.dataset = Dataset(self.image_h, self.image_w, self.image_c, self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        self.sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess_config.gpu_options.allow_growth = True

    def msra_initializer(self, kl, dl):
        stddev = math.sqrt(2. / (kl**2 * dl))
        return tf.truncated_normal_initializer(stddev=stddev)

    def cal_loss(self, logits, labels):
        labels = tf.cast(labels, tf.int32)
        return Jaccard_loss(logits, labels, num_classes=self.num_classes)

    def encoder(self, images, phase_train):
        norm1 = tf.nn.lrn(input=images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

        conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
        pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
        pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")
        pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")
        pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        return [pool4, pool4_indices, pool3_indices, pool2_indices, pool1_indices]

    def decoder(self, encode_output, phase_train):
        pool4, pool4_indices, pool3_indices, pool2_indices, pool1_indices = encode_output

        upsample4 = unpool_2d(pool4, pool4_indices, stride=[1, 2, 2, 1], scope='unpool_2d')
        conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

        upsample3 = unpool_2d(conv_decode4, pool3_indices, stride=[1, 2, 2, 1], scope='unpool_2d')
        conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

        upsample2 = unpool_2d(conv_decode3, pool2_indices, stride=[1, 2, 2, 1], scope='unpool_2d')
        conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

        upsample1 = unpool_2d(conv_decode2, pool1_indices, stride=[1, 2, 2, 1], scope='unpool_2d')
        conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")

        return conv_decode1

    def classifier(self, decode_output, labels):
        with tf.variable_scope('conv_classifier') as scope:
            kernel = _variable_with_weight_decay(name='weights',
                                                 shape=[1, 1, 64, self.num_classes],
                                                 initializer=self.msra_initializer(1, 64),
                                                 wd=0.0005)
            conv = tf.nn.conv2d(decode_output, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
            conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

        logit = conv_classifier
        loss = self.cal_loss(conv_classifier, labels)

        return loss, logit

    def inference(self, images, labels, phase_train):
        encode_output = self.encoder(images, phase_train)
        decode_output = self.decoder(encode_output, phase_train)
        return self.classifier(decode_output, labels)

    def build_graph(self, total_loss, global_step):
        lr = tf.train.exponential_decay(learning_rate=self.intial_learning_rate,
                                        global_step=global_step,
                                        decay_steps=self.num_steps_per_decay,
                                        decay_rate=self.learning_rate_decay_factor,
                                        staircase=True)
        loss_averages_op = _add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op, opt._lr

    def train(self):
        if_finetune = True if self.mode == 'finetune' else False
        finetune_ckpt = self.log_dir + self.ckpt
        startstep = 0 if not if_finetune else int(self.ckpt.split('-')[-1]) + 1

        image_filenames, label_filenames = self.dataset.get_filename_list(self.train_path)
        val_image_filenames, val_label_filenames = self.dataset.get_filename_list(self.val_path)

        with tf.Graph().as_default():
            train_data_node = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_h, self.image_w, self.image_c])
            train_labels_node = tf.placeholder(tf.int64, shape=[self.batch_size, self.image_h, self.image_w, 1])
            phase_train = tf.placeholder(tf.bool, name='phase_train')
            global_step = tf.Variable(0, trainable=False)

            images, labels = self.dataset.batch(self.batch_size, image_filenames, label_filenames)
            val_images, val_labels = self.dataset.batch(self.batch_size, val_image_filenames, val_label_filenames)
            # Build a Graph that computes the logits predictions from the inference model.
            loss, eval_prediction = self.inference(train_data_node, train_labels_node, phase_train)

            # Build a Graph that trains the model with one batch of examples and updates the model parameters.
            train_op, lr = self.build_graph(loss, global_step)
            saver = tf.train.Saver(tf.global_variables())
            summary_op = tf.summary.merge_all()

            with tf.Session(config=self.sess_config) as sess:
                # Build an initialization operation to run below.
                if if_finetune:
                    saver.restore(sess, finetune_ckpt)
                else:
                    init = tf.global_variables_initializer()
                    sess.run(init)

                # Summery placeholders
                summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
                average_pl = tf.placeholder(tf.float32)
                acc_pl = tf.placeholder(tf.float32)
                iu_pl = tf.placeholder(tf.float32)
                average_summary = tf.summary.scalar("test_average_loss", average_pl)
                acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
                iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

                for step in range(startstep, startstep + self.max_steps):
                    image_batch, label_batch = sess.run([images, labels])
                    # since we still use mini-batches in validation, still set bn-layer phase_train = True
                    feed_dict = {
                        train_data_node: image_batch,
                        train_labels_node: label_batch,
                        phase_train: True
                    }
                    start_time = time.time()

                    _, loss_value, cur_lr = sess.run([train_op, loss, lr], feed_dict=feed_dict)
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        num_examples_per_step = self.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                        # eval current training batch pre-class accuracy
                        pred = sess.run(eval_prediction, feed_dict=feed_dict)
                        per_class_acc(pred, label_batch)

                    if step % 100 == 0:
                        print("start validating.....")
                        total_val_loss = 0.0
                        hist = np.zeros((self.num_classes, self.num_classes))
                        for test_step in range(int(self.val_iter)):
                            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

                            _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                                train_data_node: val_images_batch,
                                train_labels_node: val_labels_batch,
                                phase_train: True
                            })
                            total_val_loss += _val_loss
                            hist += get_hist(_val_pred, val_labels_batch)
                        print("val loss: ", total_val_loss / self.val_iter)
                        acc_total = np.diag(hist).sum() / hist.sum()
                        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                        test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / self.val_iter})
                        acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                        iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                        print_hist_summery(hist)
                        print(" end validating.... ")

                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.add_summary(test_summary_str, step)
                        summary_writer.add_summary(acc_summary_str, step)
                        summary_writer.add_summary(iu_summary_str, step)
                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == self.max_steps + startstep:
                        checkpoint_path = os.path.join(self.log_dir, 'model')
                        saver.save(sess, checkpoint_path, global_step=step)
                        print(" model saved to log_dir")

    def test(self):
        batch_size = 1  # testing should set BATCH_SIZE = 1
        image_filenames, label_filenames = self.dataset.get_filename_list(self.test_path)

        test_data_node = tf.placeholder(tf.float32, shape=[batch_size, self.image_h, self.image_w, self.image_c])
        test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, self.image_h, self.image_w, 1])
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        loss, logits = self.inference(test_data_node, test_labels_node, phase_train)
        pred = tf.argmax(logits, axis=3)

        # get moving avg
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        with tf.Session(config=self.sess_config) as sess:
            # Load checkpoint
            saver.restore(sess, self.test_ckpt)

            images, labels = self.dataset.get_all_test_data(image_filenames, label_filenames)

            hist = np.zeros((self.num_classes, self.num_classes))

            count = 0

            for image_batch, label_batch, path in zip(images, labels, image_filenames):
                feed_dict = {
                    test_data_node: image_batch,
                    test_labels_node: label_batch,
                    phase_train: False
                }

                dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)

                # output_image to verify
                if self.output:
                    if not os.path.exists(ROOT_DIR + '/outputs/'):
                        os.mkdir(ROOT_DIR + '/outputs/')

                    image_name = str(image_filenames[count].split('/')[-1])
                    save_dir = 'outputs/' + image_name
                    writeImage(im[0], save_dir)
                    print('Prediction image %s saved to: ' % image_name + save_dir)
                    count += 1

                hist += get_hist(dense_prediction, label_batch)

            acc_total = np.diag(hist).sum() / hist.sum()
            iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
            print("acc: " + str(acc_total))
            print("mean IU: " + str(np.nanmean(iu)))


class ADDA(SegNet):
    def __init__(self, args):
        SegNet.__init__(self, args)

        self.s_source = 's'
        self.s_target = 't'
        self.s_discriminator = 'd'
        self.s_generator = 'g'
        self.src_image_path = args.train_path
        self.tar_image_path = args.da_path
        self.log_dir = args.log_dir
        self.ckpt_dir = args.log_dir + args.ckpt
        self.src_ckpt_path = args.log_dir + 'adda/src_model'
        self.tar_ckpt_path = args.log_dir + 'adda/tar_model'
        self.learning_rate_g = 1e-3
        self.learning_rate_d = 1e-3

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

            if not os.path.exists(ROOT_DIR + '/logs/adda/'):
                os.mkdir(ROOT_DIR + '/logs/adda/')

            src_saver.save(sess, self.src_ckpt_path)
            print('Source model saved to: %s' % self.src_ckpt_path)
            tar_saver.save(sess, self.tar_ckpt_path)
            print('Target model saved to: %s' % self.tar_ckpt_path)
            tmp_saver.save(sess, self.log_dir + 'adda/model.ckpt')  # rewrite the checkpoint

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

        with tf.variable_scope(self.s_source):
            src_encode_output = self.encoder(src_image, tf.constant(False))
            src_decode_output = self.decoder(src_encode_output, tf.constant(False))
            src_cls_loss, src_logits = self.classifier(src_decode_output, src_label)
        with tf.variable_scope(self.s_discriminator):
            dis_src = self.discriminator(src_encode_output[0])

        with tf.variable_scope(self.s_target):
            tar_encode_output = self.encoder(tar_image, phase_train=tf.constant(True))
            tar_decode_output = self.decoder(tar_encode_output, tf.constant(False))
            tar_cls_loss, tar_logits = self.classifier(tar_decode_output, tar_label)
        with tf.variable_scope(self.s_discriminator, reuse=True):
            dis_tar = self.discriminator(tar_encode_output[0])

        generator_loss, discriminator_loss = adversarial_loss(dis_src, dis_tar)

        var_tar = tf.trainable_variables(self.s_target)
        optim_g = tf.train.AdamOptimizer(self.learning_rate_g).minimize(generator_loss, var_list=var_tar)

        var_d = tf.trainable_variables(self.s_discriminator)
        optim_d = tf.train.AdamOptimizer(self.learning_rate_d).minimize(discriminator_loss, var_list=var_d)

        with tf.Session(config=self.sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            self.src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.s_source)
            self.tar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.s_target)

            src_saver = tf.train.Saver(self.src_vars)
            tar_saver = tf.train.Saver(self.tar_vars)

            src_saver.restore(sess, self.src_ckpt_path)
            tar_saver.restore(sess, self.tar_ckpt_path)

            for i in range(self.max_steps):
                _, d_loss_ = sess.run([optim_d, discriminator_loss])
                _, g_loss_ = sess.run([optim_g, generator_loss])

                if i % 10 == 0:
                    print("------------------------- Step: %d -------------------------" % i)
                    print('Generator loss: %.4f, Discriminator loss: %.4f' % (g_loss_, d_loss_))
                if i % 1000 == 0:
                    print("------------------------- Evaluating -------------------------")
                    pd = tf.argmax(tar_logits, axis=3)
                    _, pred = sess.run([src_logits, pd])
                    pred = pred[0]
                    pred[pred == 0] = 255
                    pred[pred == 1] = 128
                    pred[pred == 2] = 0
                    pred_image = Image.fromarray(np.uint8(pred))
                    save_path = self.log_dir + '%06d.png' % i
                    pred_image.save(save_path)
                    print("Evaluation image saved to: %s" % save_path)

                if i % 1000 == 0 or (i + 1) == self.max_steps:
                    s_path = os.path.join(self.log_dir, 'adda/src_model')
                    src_saver.save(sess, s_path, global_step=i)
                    print('Source model saved to: %s' % self.log_dir + 'adda/')
                    t_path = os.path.join(self.log_dir, 'adda/tar_model')
                    tar_saver.save(sess, t_path, global_step=i)
                    print('Target model saved to: %s' % self.log_dir + 'adda/')

    def test(self):
        self.batch_size = 1
        src_image_filenames, src_label_filenames = self.dataset.get_filename_list(self.src_image_path)
        tar_image_filenames, tar_label_filenames = self.dataset.get_filename_list(self.tar_image_path)

        src_image, src_label = self.dataset.batch(self.batch_size, src_image_filenames, src_label_filenames)
        tar_image, tar_label = self.dataset.batch(self.batch_size, tar_image_filenames, tar_label_filenames)

        with tf.variable_scope(self.s_source):
            src_encode_output = self.encoder(src_image, tf.constant(False))
            src_decode_output = self.decoder(src_encode_output, tf.constant(False))
            src_cls_loss, src_logits = self.classifier(src_decode_output, src_label)
        with tf.variable_scope(self.s_discriminator):
            dis_src = self.discriminator(src_encode_output[0])

        with tf.variable_scope(self.s_target):
            tar_encode_output = self.encoder(tar_image, phase_train=tf.constant(True))
            tar_decode_output = self.decoder(tar_encode_output, tf.constant(False))
            tar_cls_loss, tar_logits = self.classifier(tar_decode_output, tar_label)
        with tf.variable_scope(self.s_discriminator, reuse=True):
            dis_tar = self.discriminator(tar_encode_output[0])

        generator_loss, discriminator_loss = adversarial_loss(dis_src, dis_tar)

        var_tar = tf.trainable_variables(self.s_target)
        optim_g = tf.train.AdamOptimizer(self.learning_rate_g).minimize(generator_loss, var_list=var_tar)

        # var_d = tf.trainable_variables(self.s_discriminator)
        # optim_d = tf.train.AdamOptimizer(self.learning_rate_d).minimize(discriminator_loss, var_list=var_d)

        with tf.Session(config=self.sess_config) as sess:
            # self.src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.s_source)
            self.tar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.s_target)

            # src_saver = tf.train.Saver(self.src_vars)
            tar_saver = tf.train.Saver(self.tar_vars)

            # src_saver.restore(sess, self.src_ckpt_path)
            tar_saver.restore(sess, self.ckpt_dir)

            # _, d_loss_, = sess.run([optim_d, discriminator_loss])
            # _, g_loss_ = sess.run([optim_g, generator_loss])

            images, labels = self.dataset.get_all_test_data(tar_image_filenames, tar_label_filenames)

            hist = np.zeros((self.num_classes, self.num_classes))
            pred = tf.argmax(tar_logits, axis=3)

            count = 0

            for image_batch, label_batch, path in zip(images, labels, tar_image_filenames):

                dense_prediction, im = sess.run([tar_logits, pred])

                # output_image to verify
                if self.output:
                    if not os.path.exists(ROOT_DIR + '/outputs/'):
                        os.mkdir(ROOT_DIR + '/outputs/')
                    if not os.path.exists(ROOT_DIR + '/outputs/adda/'):
                        os.mkdir(ROOT_DIR + '/outputs/adda/')

                    image_name = str(tar_image_filenames[count].split('/')[-1])
                    save_dir = 'outputs/adda/' + image_name
                    writeImage(im[0], save_dir)
                    print('Prediction image %s saved to: ' % image_name + save_dir)
                    count += 1

                hist += get_hist(dense_prediction, label_batch)

            acc_total = np.diag(hist).sum() / hist.sum()
            iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
            print("acc: " + str(acc_total))
            print("mean IU: " + str(np.nanmean(iu)))
