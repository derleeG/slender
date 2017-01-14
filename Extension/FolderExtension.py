from __future__ import print_function

import random
import sys
sys.path.append('../slender')
import numpy as np
import os
import tensorflow as tf
import csv
import scipy

from slender.blob import Blob
from slender.util import scope_join_fn
from slender.producer import BaseProducer

reload(sys)
sys.setdefaultencoding('utf-8')

_ = scope_join_fn('producer')


class LocalFolderFileProducer(BaseProducer):
    def __init__(self,
                 working_dir,
                 image_dir,
                 class_names=None,
                 batch_size=64,
                 num_parallels=1):
        self.classname_path = os.path.join(working_dir, BaseProducer._CLASSNAME_NAME)
        if not class_names is None:
            np.savetxt(self.classname_path, class_names, fmt='%s')

        super(LocalFolderFileProducer, self).__init__(
                working_dir=working_dir,
                image_dir=image_dir,
                batch_size=batch_size
                )

        self.filename_list = []
        self.label_list = []

        for folder_name in os.listdir(image_dir):
            folder_path = os.path.join(image_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                if not file_name[-4:] == '.jpg':
                    continue
                file_path = os.path.join(folder_path, file_name)
                self.filename_list.append(file_path)
                self.label_list.append([0,0,0,0])

        self.num_files = len(self.filename_list)
        self.num_batches_per_epoch = self.num_files // self.batch_size
        self.num_parallels = num_parallels
        np.savetxt(os.path.join(image_dir, 'file_list.txt'), self.filename_list,  delimiter=',', fmt='%s')

    def check(self):
        print("checking")

    def blob(self):
        with tf.variable_scope(_('blob')):
            file_names = tf.convert_to_tensor(self.filename_list, dtype=tf.string)
            labels = tf.convert_to_tensor(np.array(self.label_list), dtype=tf.bool)
            filename_label_queue = BaseProducer.queue_join(
                [(file_names, labels)],
                enqueue_many=True,
            )
            filename_labels = [
                filename_label_queue.dequeue()
                for num_parallel in xrange(self.num_parallels)
            ]
            content_labels = [
                (tf.read_file(filename_label[0]), filename_label[1])
                for filename_label in filename_labels
            ]
            content_label_queue = BaseProducer.queue_join(content_labels)
            (self.contents, self.labels) = content_label_queue.dequeue_many(self.batch_size)

        return Blob(contents=self.contents, labels=self.labels)





