from Extension.CsvExtension import LocalCsvFileProducer as Producer
import Extension.LabelBinarizer as lb
#from Extension.LabelEncode import EncodeOps
from slender.processor import TestProcessor as Processor
from slender.net import SimpleNet as Net
from slender.util import latest_working_dir
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TYPE = 'price'
IMAGE_DIR = '/mnt/data/tabelog_pictures/'
WORKING_DIR = latest_working_dir('/home/derlee/Resnet_slender/')
#WORKING_DIR = '/home/derlee/Resnet_slender/2017-01-04-213504/'
#WORKING_DIR = '/home/derlee/Resnet_slender/2017-01-04-122645/'
CSV_DIR = '/mnt/data/tabelog_csv/'
RELATIVE_PATH = ['restaurant_id', 'pic_type', 'pic_name']
LABEL_OPS = lb.encode(TYPE)
FILTER_OPS = (['pic_type', 'region_name', 'night_price_2', 'restaurant_id'],
              lambda x, y, z, w: not(x == 'Dish' and hash(y)%4 == 0 and z != '0' and hash(w)%7 == 0))
RATIO = np.loadtxt('ratio.txt', delimiter=',')

BATCH_SIZE = 64
GPU_FRAC = 0.5

producer = Producer(
    image_dir=IMAGE_DIR,
    working_dir=WORKING_DIR,
    batch_size=BATCH_SIZE,
    csv_dir = CSV_DIR,
    file_path = RELATIVE_PATH,
    label_ops = LABEL_OPS,
    filter_ops = FILTER_OPS,
    class_names = lb.classnames(TYPE),
    mix_scheme = 3,
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
for i in xrange(3*producer.num_batches_per_epoch):
    a = net.eval(blob)
    values = a._dict
    p = np.concatenate((p,values['logits']),axis=0)
    y = np.concatenate((y,values['targets']),axis=0)

testSize = (-1, BATCH_SIZE, producer.num_classes)
p = p.reshape(testSize)
y = y.reshape(testSize)
p = np.sum(p, axis=1)
y = np.sum(y, axis=1)
p.reshape((-1, producer.num_classes))
y.reshape((-1, producer.num_classes))
p = 1*(p == np.max(p, axis=1, keepdims=True))
y = 1*(y == np.max(y, axis=1, keepdims=True))
c = np.matmul(y.T,p)
c = 1.0*c/np.sum(c, axis=1, keepdims=True)
acc = np.diag(c)
with open('ratio.log', 'a') as f_handle:
    f_handle.write("acc: ")
    f_handle.write(np.array_str(acc))
    f_handle.write(", ratio: ")
    f_handle.write(np.array_str(RATIO))
    f_handle.write("\n")
RATIO = RATIO/acc
RATIO = RATIO/np.sum(RATIO)
np.savetxt('ratio.txt', RATIO, delimiter=',', fmt='%s')


#np.savetxt(os.path.join(WORKING_DIR, 'predict'), p, delimiter=',', fmt='%s')
#np.savetxt(os.path.join(WORKING_DIR, 'label'), y, delimiter=',', fmt='%s')
