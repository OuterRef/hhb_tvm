#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def max_unpool_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(128, high=512, size=1))
    in_size_y  = int(np.random.randint(128, high=512, size=1))
    in_channel = int(np.random.randint(1, high=64, size=1))
    stride_x   = int(np.random.randint(1, high=3, size=1))
    stride_y   = int(np.random.randint(1, high=3, size=1))
    kernel_x   = int(np.random.randint(stride_x + 1, high=7, size=1))
    kernel_y   = int(np.random.randint(stride_y + 1, high=7, size=1))
    pad_x      = (in_size_x - kernel_x) -  int((in_size_x - kernel_x) / stride_x) * stride_x
    pad_y      = (in_size_y - kernel_y) -  int((in_size_y - kernel_y) / stride_y) * stride_y
    pad_left   = pad_right = pad_top = pad_down = 0

    if (pad_x != 0):
        pad_x      = int((in_size_x - kernel_x) / stride_x) * stride_x + stride_x - (in_size_x - kernel_x)
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    if (pad_y != 0):
        pad_y      = int((in_size_y - kernel_y) / stride_y) * stride_y + stride_y - (in_size_y - kernel_y)
        pad_top    = int(np.random.randint(0, high=pad_y, size=1))
        pad_down   = pad_y - pad_top


    out_x  = (in_size_x + pad_x - kernel_x)/stride_x + 1
    out_y  = (in_size_y + pad_y - kernel_y)/stride_y + 1

    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1)) 

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))

    t_src_in  = tensor(src_in)
    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out = fn.max_pool2d(t_src_in, (kernel_y, kernel_x), stride=(stride_y, stride_x), padding=0, return_indices=True)
    #print((pad_left, pad_right, pad_top, pad_down))
    #print((batch, in_channel, in_size_y, in_size_x))
    #print(t_src_out[1])

    t_src_uo  = fn.max_unpool2d(t_src_out[0], t_src_out[1], (kernel_y, kernel_x), stride=(stride_y, stride_x), padding=(pad_left, pad_right, pad_top, pad_down), output_size=(batch, in_channel, in_size_y, in_size_x))
    #permute nchw to nhwc
    src_in_nhwc = np.transpose(t_src_out[0], [3, 1, 0, 2])
    out_nhwc    = np.transpose(t_src_uo, [3, 1, 0, 2])
    in_indices  = np.transpose(t_src_out[1], [3, 1, 0, 2])
    #out_indices = t_src_out[1]

    src_in_1      = src_in_nhwc.flatten()
    src_out_1     = out_nhwc.flatten()
    in_indices_1  = in_indices.flatten()
    #print(batch*out_y*out_x*in_channel)

    total_size = (len(src_in_1) + len(src_out_1)) + len(in_indices_1) + 12

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(stride_y)
    para.append(stride_x)
    para.append(kernel_y)
    para.append(kernel_x)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)

    with open("max_unpool_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%di' % len(in_indices_1)), *in_indices_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)

        fp.close()

    return 0


if __name__ == '__main__':
    max_unpool_f32()
    print("end")
