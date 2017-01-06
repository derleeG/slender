
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import LabelBinarizer as LB

class_conversion = sio.loadmat('Extension/class_conversion.mat')

def classnames(s):
    if s=='price':
        str = ["".join(x) for x in class_conversion['price_name'].tolist()[0]]
        return np.array([l.encode('utf-8') for l in str])
    elif s=='food type':
        str = (["".join(x) for x in class_conversion['foodtype_name'].tolist()[0]] +
                ["".join(x) for x in class_conversion['restauranttype_name'].tolist()[0]])
        return np.array([l.encode('utf-8') for l in str])

def encode(s):
    lb = LB()
    if s=='price':
        str = ["".join(x) for x in class_conversion['night_price_name'].tolist()[0]]
        classnames =  np.array([l.encode('utf-8') for l in str])
        lb.fit(classnames)
	reduction_mat = class_conversion['price_mat']
	return (['night_price_2'],
		lambda x : np.matmul(np.sum(lb.transform([x]),0), reduction_mat)>0)
    elif s=='food type':
        str = ["".join(x) for x in class_conversion['category_name'].tolist()[0]]
        classnames =  np.array([l.encode('utf-8') for l in str])
        lb.fit(classnames)
	reduction_mat = np.concatenate((class_conversion['foodtype_mat'],
                                        class_conversion['restauranttype_mat']), axis=1)
	return (['category_1','category_2','category_3'],
		lambda x, y, z: np.matmul(np.sum(lb.transform([x, y, z]),0), reduction_mat)>0)



