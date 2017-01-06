from Extension.CsvExtension import LocalCsvFileProducer as Producer
import Extension.LabelBinarizer as lb
#from Extension.LabelEncode import EncodeOps
from slender.processor import TestProcessor as Processor
from slender.net import SimpleNet as Net
from slender.util import latest_working_dir
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TYPE = 'price'
IMAGE_DIR = '/home/derlee/tabelog_pictures/'
#WORKING_DIR = latest_working_dir('/home/derlee/Resnet_slender/')
WORKING_DIR = '/home/derlee/Resnet_slender/2017-01-04-122645/'
CSV_DIR = '/home/derlee/tabelog_csv/'
RELATIVE_PATH = ['restaurant_id', 'pic_type', 'pic_name']
LABEL_OPS = lb.encode(TYPE)
FILTER_OPS = (['pic_type', 'region_name', 'night_price_2', 'restaurant_id'],
              lambda x, y, z, w: not(x == 'Dish' and hash(y)%4 == 0 and z != '0' and hash(w)%7 != 0))
FILTER_OPS_TEST = (['pic_type', 'region_name', 'night_price_2', 'restaurant_id'],
              lambda x, y, z, w: not(x == 'Dish' and hash(y)%4 == 0 and z != '0' and hash(w)%7 == 0))


BATCH_SIZE = 128
GPU_FRAC = 0.9

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    batch_size=BATCH_SIZE,
    csv_dir = CSV_DIR,
    file_path = RELATIVE_PATH,
    label_ops = LABEL_OPS,
    filter_ops = FILTER_OPS_TEST,
    class_names = lb.classnames(TYPE),
    mix_scheme = 1,
    sample_ratio = 0.01,
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
y = np.zeros([0,producer.num_classes])
for i in xrange(producer.num_batches_per_epoch):
    print i
    a = net.eval(blob)
    values = a._dict
    p = np.concatenate((p,values['predictions']),axis=0)
    y = np.concatenate((y,values['targets']),axis=0)
np.savetxt(os.path.join(WORKING_DIR, 'predict'), p, delimiter=',', fmt='%s')
np.savetxt(os.path.join(WORKING_DIR, 'label'), y, delimiter=',', fmt='%s')
