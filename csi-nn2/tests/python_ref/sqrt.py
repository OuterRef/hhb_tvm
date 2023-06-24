#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np

def sqrt_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(32, high=64, size=1))
    in_size_y   = int(np.random.randint(32, high=64, size=1))
    in_channel  = int(np.random.randint(1, high=64, size=1))
    input_min   = int(np.random.randint(1, high=10, size=1))
    input_max   = int(np.random.randint(20, high=100, size=1))

    src_in = np.random.uniform(input_min, input_max, (batch, in_size_y, in_size_x, in_channel))

    src_in = src_in.astype(np.float32)
    src_out = np.sqrt(src_in)

    size_all = batch*in_size_y*in_size_x*in_channel
    src_in_1  = src_in.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_out_1)) + 4

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)

    with open("sqrt_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    sqrt_f32()
    print("end")
