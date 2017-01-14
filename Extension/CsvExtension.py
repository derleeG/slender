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

_ = scope_join_fn('producer')



class LocalCsvFileProducer(BaseProducer):
    class MixScheme:
        NONE = 0
        RANDOM_SINGLE = 1
        UNIFORM_SINGLE = 2
        RANDOM_MULTI = 3
        UNIFORM_MULTI = 4
        FIX_RATIO_SINGLE = 5

    def __init__(self,
                 working_dir,
                 image_dir,
                 csv_dir,
                 file_path,
                 label_ops,
                 filter_ops=None,
                 class_names=None,
                 batch_size=64,
                 num_parallels=1,
                 sample_ratio=0.1,
                 mix_scheme=MixScheme.RANDOM_SINGLE):
        # workaround to avoid imagenet style file arrangment
        self.classname_path = os.path.join(working_dir, BaseProducer._CLASSNAME_NAME)
        self.classratio_path = os.path.join(working_dir, 'class_ratio.txt')
        #class_names = np.array([l.encode('utf-8') for l in class_names])
        if not class_names is None:
            np.savetxt(self.classname_path, class_names, fmt='%s')
 
        if not os.path.isfile(self.classname_path):
            # parse class name from csv file
            class_names = list()
            label_attributes = list(label_ops[0])
            for csv_filename in os.listdir(csv_dir):
                if not csv_filename[-4:] == '.csv':
                    continue
                with open(os.path.join(csv_dir, csv_filename)) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if None not in row:
                            for attr in label_attributes:
                                class_names.append(row[attr])
            class_names = set(class_names)

            # dataset specific invalid values
            class_names.discard('C2_IS_NULL')
            class_names.discard('C3_IS_NULL')
            class_names.discard('0')
            class_names = np.sort(list(class_names))
            np.savetxt(self.classname_path, class_names, fmt='%s')

        super(LocalCsvFileProducer, self).__init__(
                working_dir=working_dir,
                image_dir=image_dir,
                batch_size=batch_size
                )

        self.filename_map = {}
        self.label_map = {}
        filter_attr, filter_op = filter_ops
        label_attr, label_op = label_ops

        for csv_filename in os.listdir(csv_dir):
            if not csv_filename[-4:] == '.csv':
                continue
            with open(os.path.join(csv_dir, csv_filename)) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if filter_op(*map(lambda i: row[i], filter_attr)):
                        continue
                    filename = os.path.join(*((image_dir,)+tuple(map(lambda i: row[i], file_path))))
                    if os.path.isfile(filename):
                        if row['restaurant_id'] in self.filename_map:
                            self.filename_map[row['restaurant_id']].append(filename)
                        else:
                            self.filename_map[row['restaurant_id']] = [filename]
                            self.label_map[row['restaurant_id']] = label_op(*map(lambda i: row[i], label_attr))

        self.num_files = sum(map(len, self.filename_map.values()))
        self.num_batches_per_epoch = self.num_files // self.batch_size
        self.num_parallels = num_parallels
        self.mix_scheme = mix_scheme
        self.sample_ratio = sample_ratio
    def check(self):
        print("checking")

    def blob(self):
        with tf.variable_scope(_('blob')):
            (file_names, labels) = zip(*[
                (file_name, label)
                for file_name, label in self.data_iterator()
            ])
            file_names = tf.convert_to_tensor(list(file_names), dtype=tf.string)
            labels = tf.convert_to_tensor(np.array(labels), dtype=tf.bool)
            filename_label_queue = BaseProducer.queue_join(
                [(file_names, labels)],
                enqueue_many=True,
                #shapes=[(),(np.shape(labels)[1],)],
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


    def data_iterator(self):
        assert len(self.filename_map.keys()) == len(self.label_map.keys())
        keys = list(self.label_map.keys())
        count = 0
        accum = np.zeros(np.shape(self.label_map[keys[0]]))
        if self.mix_scheme == LocalCsvFileProducer.MixScheme.NONE:
            for key in keys:
                for file_name in self.filename_map[key]:
                    yield file_name, self.label_map[key]
                    accum += self.label_map[key]
        elif self.mix_scheme == LocalCsvFileProducer.MixScheme.RANDOM_SINGLE:
            while count < self.num_files:
                random.shuffle(keys)
                for key in keys:
                    file_names = list(self.filename_map[key])
                    assert len(file_names) > 0
                    random.shuffle(file_names)
                    yield file_names[0], self.label_map[key]
                    accum += self.label_map[key]
                    count += 1
                    if count >= self.num_files:
                        break
        elif self.mix_scheme == LocalCsvFileProducer.MixScheme.UNIFORM_SINGLE:
            while count < self.num_files:
                random.shuffle(keys)
                for key in keys:
                    if not self.entropy_bound(accum, self.label_map[key]):
                        continue
                    file_names = list(self.filename_map[key])
                    assert len(file_names) > 0
                    random.shuffle(file_names)
                    yield file_names[0], self.label_map[key]
                    accum += self.label_map[key]
                    count += 1
                    if count >= self.num_files:
                        break
        elif self.mix_scheme == LocalCsvFileProducer.MixScheme.RANDOM_MULTI:
            while count < self.num_files*3:
                random.shuffle(keys)
                for key in keys:
                    file_names = list(self.filename_map[key])
                    assert len(file_names) > 0
                    while len(file_names) < self.batch_size:
                        file_names = file_names*2
                    random.shuffle(file_names)
                    file_names = file_names[0:self.batch_size]
                    for repeat in xrange(self.batch_size):
                        yield file_names[repeat], self.label_map[key]
                        accum += self.label_map[key]
                        count += 1
                    if count >= self.num_files*3:
                        break
        elif self.mix_scheme == LocalCsvFileProducer.MixScheme.UNIFORM_MULTI:
            while count < self.num_files:
                random.shuffle(keys)
                for key in keys:
                    if not self.entropy_bound(accum, self.label_map[key]):
                        continue
                    file_names = list(self.filename_map[key])
                    assert len(file_names) > 0
                    while len(file_names) < self.batch_size:
                        file_names = file_names*2
                    random.shuffle(file_names)
                    file_names = file_names[0:self.batch_size]
                    for repeat in xrange(self.batch_size):
                        yield file_names[repeat], self.label_map[key]
                        accum += self.label_map[key]
                        count += 1
                    if count >= self.num_files:
                        break
        elif self.mix_scheme == LocalCsvFileProducer.MixScheme.FIX_RATIO_SINGLE:
            while count < self.num_files:
                random.shuffle(keys)
                for key in keys:
                    if not self.ratio_bound(accum, self.label_map[key]):
                        continue
                    file_names = list(self.filename_map[key])
                    assert len(file_names) > 0
                    random.shuffle(file_names)
                    yield file_names[0], self.label_map[key]
                    accum += self.label_map[key]
                    count += 1
                    if count >= self.num_files:
                        break

        print("Image class ratio")
        print(accum)
        np.savetxt(self.classratio_path, accum, fmt='%s')

    def ratio_bound(self, accum, label):
        if random.random() < 0.1:
            return True
        else:
            stable = accum + 1e-8
            old_ratio = stable / np.sum(stable)
            new_ratio = stable + label
            new_ratio = new_ratio / np.sum(new_ratio)
            old_en = scipy.stats.entropy(old_ratio, self.sample_ratio)
            new_en = scipy.stats.entropy(new_ratio, self.sample_ratio)
            return new_en <= old_en

    def entropy_bound(self, accum, label):
        if random.random() < self.sample_ratio:
            return True
        else:
            stable = accum + 1e-8
            old_ratio = stable / np.sum(stable)
            new_ratio = stable + label
            new_ratio = new_ratio / np.sum(new_ratio)
            old_en =  np.sum(-old_ratio*np.log(old_ratio))
            new_en =  np.sum(-new_ratio*np.log(new_ratio))
            return new_en >= old_en


