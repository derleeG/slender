from Extension.FolderExtension import LocalFolderFileProducer as Producer
import Extension.LabelBinarizer as lb
from slender.processor import TestProcessor as Processor
from slender.net import SimpleNet as Net
from slender.util import latest_working_dir
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TYPE = 'price'
IMAGE_DIR = '/home/derlee/dish_photo/'
#WORKING_DIR = latest_working_dir('/home/derlee/Resnet_slender/')
#WORKING_DIR = '/home/derlee/Resnet_slender/2017-01-04-213504/'
WORKING_DIR = '/home/derlee/Resnet_slender/2017-01-04-122645/'

BATCH_SIZE = 64
GPU_FRAC = 0.5

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    batch_size=BATCH_SIZE,
    class_names = lb.classnames(TYPE),
)

processor = Processor()
net = Net(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    gpu_frac=GPU_FRAC,
)

blob = producer.blob().func(processor.preprocess).func(net.forward)
net.init()
p = np.zeros([0,producer.num_classes])
for i in xrange(producer.num_batches_per_epoch):
    a = net.eval(blob)
    values = a._dict
    p = np.concatenate((p,values['logits']),axis=0)

p = 1*(p == np.max(p, axis=1, keepdims=True))

np.savetxt(os.path.join(IMAGE_DIR, 'predict'), p, delimiter=',', fmt='%s')
