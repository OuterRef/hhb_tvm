#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
from torch import tensor
from torch.nn import functional as fn

def depthwise_convolution_relu6_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(64, high=128, size=1))
    in_size_y  = int(np.random.randint(64, high=128, size=1))
    in_channel = int(np.random.randint(2, high=24, size=1))
    stride_x   = int(np.random.randint(1, high=3, size=1))
    stride_y   = int(np.random.randint(1, high=3, size=1))
    kernel_x   = int(np.random.randint(stride_x + 1, high=7, size=1))
    kernel_y   = int(np.random.randint(stride_y + 1, high=7, size=1))
    dilation_x = int(np.random.randint(1, high=5, size=1))
    dilation_y = int(np.random.randint(1, high=5, size=1))
    kernel_x_t = kernel_x + (kernel_x - 1) * (dilation_x - 1)
    kernel_y_t = kernel_y + (kernel_y - 1) * (dilation_y - 1)
    pad_left   = pad_right = pad_top = pad_down = 0

    pad_x      = (in_size_x - kernel_x_t) -  int((in_size_x - kernel_x_t) / stride_x) * stride_x
    if(pad_x !=0):
        pad_left   = int(np.random.randint(0, high=pad_x, size=1))
        pad_right  = pad_x - pad_left

    pad_y      = (in_size_y - kernel_y_t) -  int((in_size_y - kernel_y_t) / stride_y) * stride_y
    if(pad_y != 0):
        pad_top    = int(np.random.randint(0, high=pad_y, size=1))
        pad_down   = pad_y - pad_top
    zero_point1 = int(np.random.randint(-3, high=3, size=1))
    std1        = int(np.random.randint(1, high=3, size=1))
    zero_point2 = int(np.random.randint(-3, high=3, size=1))
    std2        = int(np.random.randint(1, high=3, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))
    weight = np.random.normal(zero_point2, std2, (in_channel, 1, kernel_y, kernel_x))
    bias   = np.random.normal(zero_point3, std3, in_channel)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)
    bias   = bias.astype(np.float32)

    t_src_in  = tensor(src_in)
    t_weight  = tensor(weight)
    t_bias    = tensor(bias)
    t_src_in  = fn.pad(t_src_in, (pad_left, pad_right, pad_top, pad_down), 'constant', 0)
    t_src_out = fn.conv2d(t_src_in, t_weight, bias=t_bias, stride=(stride_y, stride_x), padding=0, dilation=(dilation_y, dilation_x), groups=in_channel)
    t_src_out = fn.relu6(t_src_out).numpy()

    out_size_x = np.shape(t_src_out)[3]
    out_size_y = np.shape(t_src_out)[2]
    out_channel = np.shape(t_src_out)[1]

    src_in_1  = src_in.flatten()
    weight_1  = weight.flatten()
    src_out_1 = t_src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(weight_1) + len(bias) + 17

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y) #height
    para.append(in_size_x) #width
    para.append(stride_y)
    para.append(stride_x)
    para.append(kernel_y)
    para.append(kernel_x)
    para.append(pad_left)
    para.append(pad_right)
    para.append(pad_top)
    para.append(pad_down)
    para.append(out_channel)
    para.append(dilation_y)
    para.append(dilation_x)
    para.append(out_size_y) #height
    para.append(out_size_x) #width


    with open("depthwise_convolution_relu6_nchw_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(bias)), *bias)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    depthwise_convolution_relu6_f32()
    print("end")
