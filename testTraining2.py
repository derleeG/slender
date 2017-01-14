from Extension.CsvExtension import LocalCsvFileProducer as Producer
import Extension.LabelBinarizer as lb
#from Extension.LabelEncode import EncodeOps
from slender.processor import TrainProcessor as Processor
from slender.net import TrainNet as Net
from slender.util import new_working_dir
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TYPE = 'price'
IMAGE_DIR = '/mnt/data/tabelog_pictures/'
WORKING_DIR = new_working_dir('/home/derlee/Resnet_slender/')
CSV_DIR = '/mnt/data/tabelog_csv/'
RELATIVE_PATH = ['restaurant_id', 'pic_type', 'pic_name']
LABEL_OPS = lb.encode(TYPE)
FILTER_OPS = (['pic_type', 'region_name', 'night_price_2', 'restaurant_id'],
              lambda x, y, z, w: not(x == 'Dish' and hash(y)%4 == 0 and z != '0' and hash(w)%7 != 0))
LEARNING_RATE = 0.1

RATIO = np.loadtxt('ratio.txt', delimiter=',')

BATCH_SIZE = 64
GPU_FRAC = 0.5
NUM_TRAIN_EPOCHS = 5
NUM_DECAY_EPOCHS = 0.5

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    batch_size=BATCH_SIZE,
    csv_dir = CSV_DIR,
    file_path = RELATIVE_PATH,
    label_ops = LABEL_OPS,
    filter_ops = FILTER_OPS,
    class_names = lb.classnames(TYPE),
    mix_scheme = 5,
    sample_ratio = RATIO,
)

processor = Processor()
net = Net(
    working_dir=WORKING_DIR,
    num_classes=producer.num_classes,
    learning_rate_decay_steps=NUM_DECAY_EPOCHS * producer.num_batches_per_epoch,
    gpu_frac=GPU_FRAC,
    learning_rate=LEARNING_RATE,
)

blob = producer.blob().func(processor.preprocess).func(net.forward)
net.eval(NUM_TRAIN_EPOCHS * producer.num_batches_per_epoch, save_interval_secs=600)

