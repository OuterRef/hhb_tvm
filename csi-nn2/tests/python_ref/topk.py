#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def topk_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=5, size=1))
    in_shape = []
    for i in range(0, in_dim - 1):
        in_shape.append(int(np.random.randint(16, high=64, size=1)))

    # input Tensor with last dimension at least k
    last_dim = int(np.random.randint(16, high=64, size=1))
    in_shape.append(last_dim)

    topk = int(np.random.randint(1, high=last_dim + 1, size=1))

    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    src_in = src_in.astype(np.float32)

    out_calcu = tf.math.top_k(src_in, k = topk)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    values_data = src_out[0]
    indices_data = src_out[1]

    src_in_1  = src_in.ravel('C')
    # src_out_1 = src_out.flatten()
    values_out  = values_data.flatten()
    indices_out = indices_data.flatten()

    total_size = (len(src_in_1) + len(values_out) + len(indices_out)) + 2 + in_dim

    para.append(total_size)
    para.append(topk)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    print(para)

    with open("topk_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(values_out)), *values_out)
        fp.write(data)
        data = struct.pack(('%di' % len(indices_out)), *indices_out)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    topk_f32()
    print("end")
