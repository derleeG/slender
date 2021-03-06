import abc
import os
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1

from .blob import Blob
from .util import scope_join_fn

slim = tf.contrib.slim

_ = scope_join_fn('net')

_NET = resnet_v1.resnet_v1_50
_SIZE = resnet_v1.resnet_v1.default_image_size
_ARG_SCOPE_FN = resnet_v1.resnet_arg_scope

_NET_SCOPE = _NET.__name__
__ = scope_join_fn(_NET_SCOPE)
_CKPT_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)),
    os.pardir,
    'model',
    _NET_SCOPE + '.ckpt',
)


class BaseNet(object):
    __metaclass__ = abc.ABCMeta

    _METRIC_ATTRS = [
#        'conf_loss',
#        'train_ratio',
#        'closs',
        'loss',
        'total_loss',
        'accuracy',
    ]
    _OUTPUT_FLATTENED_ATTRS = [
        'targets',
        'logits',
        'predictions',
    ]

    @staticmethod
    def get_scope_set(scopes=[''], collection=tf.GraphKeys.VARIABLES):
        return set().union(*[slim.get_variables(scope, collection=collection) for scope in scopes])

    def __init__(self,
                 working_dir,
                 num_classes,
                 is_training,
                 weight_decay=1e-3,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        self.working_dir = working_dir
        self.num_classes = num_classes
        self.arg_scope = _ARG_SCOPE_FN(
            is_training=is_training,
            weight_decay=weight_decay,
        )
        self.session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac),
            #gpu_options=tf.GPUOptions(allow_growth = True),
            log_device_placement=log_device_placement,
        )
        self.graph = tf.get_default_graph()

        tf.logging.set_verbosity(verbosity)

    @abc.abstractmethod
    def _forward(self):
        pass

    def forward(self, blob):
        self.images = blob.images
        self.labels = blob.labels

        with slim.arg_scope(self.arg_scope):
            (net, self.end_points) = _NET(
                self.images,
                global_pool=False,
                scope=_NET_SCOPE,
            )

            with tf.variable_scope(_('forward')):
                self.feats = net = tf.reduce_mean(
                    net,
                    (1, 2),
                    keep_dims=True,
                    name='feats',
                )
                self.logits = net = slim.conv2d(
                    net,
                    self.num_classes,
                    (1, 1),
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logits',
                )
                self.predictions = net = slim.softmax(
                    net,
                    scope='predictions',
                )
#                self.conf = slim.conv2d(
#                    self.feats,
#                    1,
#                    (1, 1),
#                    activation_fn=tf.nn.sigmoid,
#                    normalizer_fn=None,
#                    scope='confidence',
#                )
#                self.conf_flattened = tf.squeeze(self.conf, (1, 2, 3))


                self.predictions_flattened = tf.squeeze(self.predictions, (1, 2))
                #self.targets = tf.one_hot(self.labels, self.num_classes)                   #//
                self.targets = tf.to_float(self.labels)                                     #//
                self.loss = slim.losses.log_loss(self.predictions_flattened, self.targets, weight=self.num_classes)
#                losses = -self.targets*tf.log(self.predictions_flattened + 1e-7) - (1-self.targets)*tf.log(1 - self.predictions_flattened + 1e-7)
#                losses = tf.reduce_sum(losses, reduction_indices=1)
#                self.closs = tf.reduce_mean(losses)
#                inlier = tf.to_float(losses < self.closs)
#                self.train_ratio = tf.reduce_mean(inlier)
#                self.loss = tf.reduce_sum(losses*inlier)/tf.reduce_sum(inlier)
                
#                self.conf_loss = slim.losses.log_loss(self.conf_flattened, inlier)
                self.total_loss = slim.losses.get_total_loss() #+ self.loss 

                self.predicted_labels = tf.argmax(self.predictions_flattened, 1)
                self.predicted_labels = tf.one_hot(self.predicted_labels, self.num_classes) #//
                #self.accuracy = slim.metrics.accuracy(self.predicted_labels, self.labels)  #//
                self.accuracy = tf.reduce_mean(self.predicted_labels*tf.to_float(self.labels))
                self.accuracy = self.accuracy*self.num_classes

                self.metric_names = BaseNet._METRIC_ATTRS
                self.metric_values = map(self.__getattribute__, BaseNet._METRIC_ATTRS)

                self.global_step = slim.get_or_create_global_step()
                self._forward()

                blob = Blob(**{
                    attr: slim.flatten(self.__getattribute__(attr), scope='{}_flattened'.format(attr))
                    for attr in BaseNet._OUTPUT_FLATTENED_ATTRS
                })

        return blob

    @abc.abstractmethod
    def eval(self):
        pass


class TrainNet(BaseNet):
    _SCOPES_TO_FREEZE = [
        __('conv1'),
        __('block1'),
        __('block2'),
    ]

    def __init__(self,
                 working_dir,
                 num_classes,
                 learning_rate=1.0,
                 learning_rate_decay_steps=None,
                 learning_rate_decay_rate=0.5,
                 weight_decay=1e-3,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super(TrainNet, self).__init__(
            working_dir=working_dir,
            num_classes=num_classes,
            is_training=True,
            weight_decay=weight_decay,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate

    def _forward(self):
        self.learning_rate = tf.constant(
            self.learning_rate,
            dtype=tf.float32,
            name='learning_rate',
        )

        if self.learning_rate_decay_steps is not None:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=self.learning_rate_decay_steps,
                decay_rate=self.learning_rate_decay_rate,
                staircase=True,
                name='decaying_learning_rate',
            )

        all_var_set = BaseNet.get_scope_set()
        trainvable_var_set = BaseNet.get_scope_set(collection=tf.GraphKeys.TRAINABLE_VARIABLES)

        self.vars_to_restore = all_var_set - BaseNet.get_scope_set([_('forward')])
        self.vars_to_train = (all_var_set - BaseNet.get_scope_set(TrainNet._SCOPES_TO_FREEZE)) & trainvable_var_set
        self.vars_to_save = all_var_set

        self.metric_names = map('train/'.__add__, self.metric_names)
        self.summary_ops = map(tf.scalar_summary, self.metric_names, self.metric_values)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1.0)
        self.train_op = slim.learning.create_train_op(
            self.total_loss,
            self.optimizer,
            variables_to_train=self.vars_to_train,
        )
        (assign_op, assign_feed_dict) = slim.assign_from_checkpoint(_CKPT_PATH, list(self.vars_to_restore))
        init_op = tf.initialize_variables(set(tf.all_variables()) - self.vars_to_restore)
        self.init_op = tf.group(assign_op, init_op)
        self.init_feed_dict = assign_feed_dict
        self.saver = tf.train.Saver(self.vars_to_save)

    def eval(self,
             number_of_steps,
             log_every_n_steps=1,
             save_summaries_secs=10,
             save_interval_secs=600):

        slim.learning.train(
            self.train_op,
            self.working_dir,
            log_every_n_steps=log_every_n_steps,
            number_of_steps=number_of_steps,
            init_op=self.init_op,
            init_feed_dict=self.init_feed_dict,
            save_summaries_secs=save_summaries_secs,
            saver=self.saver,
            save_interval_secs=save_interval_secs,
            session_config=self.session_config,
        )


class TestNet(BaseNet):
    def __init__(self,
                 working_dir,
                 num_classes,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super(TestNet, self).__init__(
            working_dir=working_dir,
            num_classes=num_classes,
            is_training=False,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

    def _forward(self):
        self.metric_names = map('test/'.__add__, self.metric_names)
        self.metric_value_updates = map(slim.metrics.streaming_mean, self.metric_values)
        (self.metric_values, self.metric_updates) = slim.metrics.aggregate_metrics(*self.metric_value_updates)
        self.summary_ops = map(tf.scalar_summary, self.metric_names, self.metric_values)

    def eval(self,
             num_steps,
             eval_interval_secs=300,
             timeout=600):

        slim.evaluation.evaluation_loop(
            '',
            self.working_dir,
            os.path.join(self.working_dir, 'test'),
            num_evals=num_steps,
            eval_op=self.metric_updates,
            eval_interval_secs=eval_interval_secs,
            session_config=self.session_config,
            #timeout=timeout,
        )


class SimpleNet(BaseNet):
    def __init__(self,
                 working_dir,
                 num_classes,
                 gpu_frac=1.0,
                 log_device_placement=False,
                 verbosity=tf.logging.INFO):

        super(SimpleNet, self).__init__(
            working_dir=working_dir,
            num_classes=num_classes,
            is_training=False,
            gpu_frac=gpu_frac,
            log_device_placement=log_device_placement,
            verbosity=verbosity,
        )

    def _forward(self):
        self.vars_to_restore = BaseNet.get_scope_set()

        (assign_op, assign_feed_dict) = slim.assign_from_checkpoint(tf.train.latest_checkpoint(self.working_dir), list(self.vars_to_restore))
        self.init_op = assign_op
        self.init_feed_dict = assign_feed_dict

    def init(self):
        self.sess = tf.Session(
            config=self.session_config,
        )
        self.sess.run(self.init_op, feed_dict=self.init_feed_dict)
        tf.train.start_queue_runners(self.sess)

    def eval(self, blob, feed_dict=None):
        blob_val = blob.eval(self.sess, feed_dict=feed_dict)
        return blob_val
