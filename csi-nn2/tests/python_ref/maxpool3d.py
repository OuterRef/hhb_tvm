#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def maxpool3d_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    channel    = int(np.random.randint(2, high=6, size=1))
    in_depth  = int(np.random.randint(32, high=64, size=1))
    in_height = int(np.random.randint(32, high=64, size=1))
    in_width  = int(np.random.randint(32, high=64, size=1))

    stride_d   = int(np.random.randint(1, high=4, size=1))
    stride_h   = int(np.random.randint(1, high=4, size=1))
    stride_w   = int(np.random.randint(1, high=4, size=1))

    kernel_d   = int(np.random.randint(stride_d, high=9, size=1))
    kernel_h   = int(np.random.randint(stride_h, high=9, size=1))
    kernel_w   = int(np.random.randint(stride_w, high=9, size=1))

    pad_left  = pad_right = 0
    pad_top   = pad_down  = 0
    pad_front = pad_back  = 0


    pad_w      = (in_width - kernel_w) -  int((in_width - kernel_w) / stride_w) * stride_w
    if(pad_w !=0):
        pad_w      = int((in_width - kernel_w) / stride_w) * stride_w + stride_w - (in_width - kernel_w)
        pad_left   = int(np.random.randint(0, high=pad_w, size=1))
        pad_right  = pad_w - pad_left

    pad_h      = (in_height - kernel_h) -  int((in_height - kernel_h) / stride_h) * stride_h
    if(pad_h !=0):
        pad_h      = int((in_height - kernel_h) / stride_h) * stride_h + stride_h - (in_height - kernel_h)
        pad_top   = int(np.random.randint(0, high=pad_h, size=1))
        pad_down  = pad_h - pad_top

    pad_d      = (in_depth - kernel_d) -  int((in_depth - kernel_d) / stride_d) * stride_d
    if(pad_d !=0):
        pad_d      = int((in_depth - kernel_d) / stride_d) * stride_d + stride_d - (in_depth - kernel_d)
        pad_front   = int(np.random.randint(0, high=pad_d, size=1))
        pad_back  = pad_d - pad_front

    zero_point = int(np.random.randint(-8, high=8, size=1))
    std        = int(np.random.randint(1, high=3, size=1))

    src_in = np.random.normal(zero_point, std, (batch, channel, in_depth, in_height, in_width))

    t_src_in  = tensor(src_in)
    t_src_in1  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down, pad_front, pad_back), 'constant', 0)

    t_src_out = fn.max_pool3d(t_src_in1, kernel_size=(kernel_d, kernel_h, kernel_w), stride=(stride_d, stride_h, stride_w)).numpy()


    out_depth  = np.shape(t_src_out)[2]
    out_height = np.shape(t_src_out)[3]
    out_width  = np.shape(t_src_out)[4]

    src_in_1  = t_src_in.flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 20

    para.append(total_size)
    para.append(batch)
    para.append(channel)
    para.append(in_depth)
    para.append(in_height)
    para.append(in_width)
    para.append(stride_d)
    para.append(stride_h)
    para.append(stride_w)
    para.append(kernel_d)
    para.append(kernel_h)
    para.append(kernel_w)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)
    para.append(pad_front)
    para.append(pad_back)
    para.append(out_depth)
    para.append(out_height)
    para.append(out_width)
    print(para)

    with open("maxpool3d_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    maxpool3d_f32()
    print("end")
