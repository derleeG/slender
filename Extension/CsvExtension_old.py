from __future__ import print_function

import sys
sys.path.append('../slender')
import numpy as np
import os
import tensorflow as tf
import csv

from .blob import Blob
from .util import scope_join_fn

_ = scope_join_fn('producer')

class CsvMeta(object):
    CLASSNAMES_FILENAME = 'class_names.txt'

    @staticmethod
    def train(image_dir,
              csv_dir,
              working_dir,
              label_attribute):
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir)

        class_names = list()
        label_attributes = prob_list(label_attribute)
        for csv_filename in os.listdir(csv_dir):
            with open(os.path.join(csv_dir, csv_filename)) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if None not in row:
                        for attr in label_attributes:
                            class_names.append(row[attr])
        class_names = set(class_names)
        class_names.discard('C2_IS_NULL')
        class_names.discard('C3_IS_NULL')
        class_names.discard('0')
        class_names = list(class_names)
        np.savetxt(os.path.join(working_dir, Meta.CLASSNAMES_FILENAME), class_names, delimiter=',', fmt='%s')
        return Meta(working_dir=working_dir, class_names=class_names)

    @staticmethod
    def test(working_dir):
        classnames_path = os.path.join(working_dir, Meta.CLASSNAMES_FILENAME)
        if os.path.isfile(classnames_path):
            class_names = np.loadtxt(classnames_path, dtype=np.str, delimiter=',')
        return Meta(working_dir=working_dir, class_names=class_names)

    def __init__(self, working_dir, class_names):
        self.working_dir = working_dir
        self.class_names = class_names


class CsvFileProducer(BaseProducer):
    CAPACITY = 512
    NUM_TRAIN_INPUTS = 8
    NUM_TEST_INPUTS = 1
    SUBSAMPLE_SIZE = 32

    def __init__(self,
                 capacity=CAPACITY,
                 num_train_inputs=NUM_TRAIN_INPUTS,
                 num_test_inputs=NUM_TEST_INPUTS,
                 subsample_size=SUBSAMPLE_SIZE
                 ):

        self.capacity = capacity
        self.num_train_inputs = num_train_inputs
        self.num_test_inputs = num_test_inputs
        self.subsample_size = subsample_size

    def _blob(self,
              image_dir,
              csv_dir,
              file_path,
              label_ops,
              filter_ops=None,
              num_inputs=1,
              check=False,
              shuffle=False
              ):

        filename_list = list()
        #classname_list = list()
        label_list = list()
        filter_attr, filter_op = filter_ops
        label_attr, label_op = label_ops

        for csv_filename in os.listdir(csv_dir):
            with open(os.path.join(csv_dir, csv_filename)) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if filter_op(*map(lambda i: row[i], filter_attr)):
                        continue
                    filename = os.path.join(*((image_dir,)+tuple(map(lambda i: row[i], file_path))))
                    if os.path.isfile(filename):
                        filename_list.append(filename)
                        #classname_list.append(row[label_ops])
                        label_list.append(label_op(*map(lambda i: row[i], label_attr)))

        #label_list = map(ResNet.META.class_names.index, classname_list)

        print('Data set size: %d' % len(filename_list))
        #np.savetxt(os.path.join(ResNet.META.working_dir, "filenames.txt"), zip(filename_list, label_list), delimiter=',', fmt='%s')

        images = list()
        labels = list()
        for num_input in xrange(num_inputs):
            if shuffle:
                perm = np.random.permutation(len(filename_list))
                filename_list = map(filename_list.__getitem__, perm)
                label_list = map(label_list.__getitem__, perm)

            label_list = np.array(label_list)
            filename_queue = self.get_queue_enqueue(filename_list, dtype=tf.string, shape=(), auto=True)[0]
            (key, value) = tf.WholeFileReader().read(filename_queue)
            image = tf.to_float(tf.image.decode_jpeg(value))

            label_queue = self.get_queue_enqueue(label_list, dtype=tf.bool, shape=(len(ResNet.META.class_names),), auto=True)[0]
            label = label_queue.dequeue()
            images.append(image)
            labels.append(label)

        return Blob(images=images, labels=labels)

    def trainBlob(self,
                  image_dir,
                  csv_dir,
                  file_path,
                  label_attribute,
                  filter_ops=None,
                  check=True,
                  ):
        return self._blob(
            image_dir,
            csv_dir,
            file_path,
            label_attribute,
            filter_ops,
            num_inputs=self.num_train_inputs,
            check=check,
            shuffle=True)

    def testBlob(self,
                 image_dir,
                 csv_dir,
                 file_path,
                 label_attribute,
                 filter_ops=None,
                 check=True,
                 ):
        return self._blob(
            image_dir,
            csv_dir,
            file_path,
            label_attribute,
            filter_ops,
            num_inputs=self.num_test_inputs,
            check=check,
            shuffle=True)

    def kwargs(self):
        return dict()
