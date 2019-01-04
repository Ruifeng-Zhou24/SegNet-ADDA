import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
import skimage
import skimage.io

class Dataset():

    def __init__(self, image_h, image_w, image_c, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
        self.IMAGE_HEIGHT = image_h
        self.IMAGE_WIDTH = image_w
        self.IMAGE_DEPTH = image_c

        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    def dataset_reader_seq(self, filename_queue, seq_length):
      image_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[0])
      label_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[1])

      image_seq = []
      label_seq = []
      for im ,la in zip(image_seq_filenames, label_seq_filenames):
        imageValue = tf.read_file(tf.squeeze(im))
        labelValue = tf.read_file(tf.squeeze(la))
        image_bytes = tf.image.decode_png(imageValue)
        label_bytes = tf.image.decode_png(labelValue)
        image = tf.cast(tf.reshape(image_bytes, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH)), tf.float32)
        label = tf.cast(tf.reshape(label_bytes, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1)), tf.int64)
        image_seq.append(image)
        label_seq.append(label)
      return image_seq, label_seq

    def dataset_reader(self, filename_queue):

      image_filename = filename_queue[0]
      label_filename = filename_queue[1]

      imageValue = tf.read_file(image_filename)
      labelValue = tf.read_file(label_filename)

      image_bytes = tf.image.decode_png(imageValue)
      label_bytes = tf.image.decode_png(labelValue)

      image = tf.reshape(image_bytes, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH))
      label = tf.reshape(label_bytes, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1))

      return image, label

    def get_filename_list(self, path):
      fd = open(path)
      image_filenames = []
      label_filenames = []
      filenames = []
      for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
      return image_filenames, label_filenames

    def batch(self, batch_size, image_filenames, label_filenames):
        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, label_filenames))
        batch = dataset.map(self._read_image_label).batch(batch_size).repeat()
        shuffle_batch = batch.shuffle(int(0.4 * len(image_filenames)))
        return shuffle_batch.make_one_shot_iterator().get_next()

    def _read_image_label(self, image_filename, label_filename):
        imageValue = tf.read_file(image_filename)
        labelValue = tf.read_file(label_filename)

        image_bytes = tf.image.decode_png(imageValue)
        label_bytes = tf.image.decode_png(labelValue)

        image = tf.reshape(image_bytes, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH))
        image = tf.cast(image, tf.float32)
        label = tf.reshape(label_bytes, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1))

        return image, label

    def get_all_test_data(self, im_list, la_list):
      images = []
      labels = []
      for im_filename, la_filename in zip(im_list, la_list):
        im = np.array(skimage.io.imread(im_filename), np.float32)
        im = im[np.newaxis]
        la = skimage.io.imread(la_filename)
        la = la[np.newaxis]
        la = la[...,np.newaxis]
        images.append(im)
        labels.append(la)
      return images, labels
