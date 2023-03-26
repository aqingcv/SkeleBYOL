import numpy as np
import random
from math import sin, cos
import math

transform_order = {
    'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
}

 #     C, T, V, M = data_numpy.shape
def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R) # TVMC * CC 
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy

def random_rotate(seq):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        temp = np.matmul(seq, R)
        return temp

    new_seq = seq.copy()
    # C, T, V, M -> T, V, M, C
    new_seq = np.transpose(new_seq, (1, 2, 3, 0))
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    new_seq = np.transpose(new_seq, (3, 0, 1, 2))

    return new_seq


def random_spatial_flip(seq, p=0.5):
    if random.random() < p:
        # Do the left-right transform C,T,V,M
        index = transform_order['ntu']
        trans_seq = seq[:, :, index, :]
        return trans_seq
    else:
        return seq

def gaus_noise(data_numpy, mean= 0, std=0.01, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        return temp + noise
    else:
        return data_numpy