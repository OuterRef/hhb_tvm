#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def avgpool2d_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(64, high=128, size=1))
    in_size_y  = int(np.random.randint(64, high=128, size=1))
    in_channel = int(np.random.randint(1, high=64, size=1))
    stride_x   = int(np.random.randint(1, high=3, size=1))
    stride_y   = int(np.random.randint(1, high=3, size=1))
    kernel_x   = int(np.random.randint(stride_x, high=7, size=1))
    kernel_y   = int(np.random.randint(stride_y, high=7, size=1))
    include_pad  = int(np.random.randint(0, high=2, size=1))    # 0: false  1: true

    pad_left   = pad_right = pad_top = pad_down = 0
    pad_x      = (in_size_x - kernel_x) -  int((in_size_x - kernel_x) / stride_x) * stride_x
    if(pad_x !=0):
        pad_x      = int((in_size_x - kernel_x) / stride_x) * stride_x + stride_x - (in_size_x - kernel_x)
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    pad_y      = (in_size_y - kernel_y) -  int((in_size_y - kernel_y) / stride_y) * stride_y
    if(pad_y != 0):
        pad_y      = int((in_size_y - kernel_y) / stride_y) * stride_y + stride_y - (in_size_y - kernel_y)
        pad_top    = int(np.random.randint(0, high=pad_y, size=1))
        pad_down   = pad_y - pad_top

    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))

    t_src_in  = tensor(src_in)
    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out = fn.avg_pool2d(t_src_in, (kernel_y, kernel_x), stride=(stride_y, stride_x), padding=0, count_include_pad = True if include_pad else False).numpy()

    #permute nchw to nhwc
    src_in_nhwc = np.transpose(src_in, [0, 2, 3, 1])
    out_nhwc    = np.transpose(t_src_out, [0, 2, 3, 1])

    out_height = np.shape(out_nhwc)[1]
    out_width  = np.shape(out_nhwc)[2]

    src_in_1  = src_in_nhwc.flatten()
    src_out_1 = out_nhwc.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 15

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
    para.append(out_height)
    para.append(out_width)
    para.append(include_pad)
    print(para)
    print(len(src_out_1))

    with open("averagepool_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    avgpool2d_f32()
    print("end")
