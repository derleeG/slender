
import scipy.io as sio
import numpy as np


class_conversion = sio.loadmat('class_conversion.mat')

def reduced_classnames(s):
    if s=='price':
        return ["".join(x) for x in class_conversion['price_name'].tolist()[0]]
    elif s=='food type':
        return (["".join(x) for x in class_conversion['foodtype_name'].tolist()[0]] +
                ["".join(x) for x in class_conversion['restauranttype_name'].tolist()[0]])

def reduction_matrix(s):
    if s=='price':
        return class_conversion['price_mat']
    elif s=='food type':
        return np.concatenate((class_conversion['foodtype_mat'],
                              class_conversion['restauranttype_mat']), axis=1)



