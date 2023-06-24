#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
from tensorflow import keras as tk

def threshold_relu_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(128, high=512, size=1))
    in_size_y  = int(np.random.randint(128, high=512, size=1))
    in_channel = int(np.random.randint(1, high=64, size=1))
    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))
    src_in = src_in.astype(np.float32)
    thetas = np.random.random(1)
    thetas = thetas.astype(np.float32)

    src_in_k  = tk.Input(shape=(in_size_y, in_size_x, in_channel), batch_size=batch, dtype=float)
    out_calcu = tk.layers.ThresholdedReLU(theta=thetas[0])(src_in_k)
    keras_model = tk.models.Model(src_in_k, out_calcu)
    src_out = keras_model.predict(src_in)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)

    with open("threshold_relu_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(thetas)), *thetas)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    threshold_relu_f32()
    print("end")
