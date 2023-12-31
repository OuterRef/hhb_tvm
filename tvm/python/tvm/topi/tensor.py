# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""
from __future__ import absolute_import as _abs
from . import cpp
from .. import te, tir
from .utils import get_const_tuple, prod


def elemwise_sum(xs):
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.te.Tensor
        Input arguments.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.elemwise_sum(xs)


def full(shape, dtype, fill_value):
    """Fill tensor with fill_value

    Parameters
    ----------
    shape : tuple
        Input tensor shape.
    dtype : str
        Data type
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full(shape, dtype, fill_value)


def full_like(x, fill_value):
    """Construct a tensor with same shape as input tensor,
       then fill tensor with fill_value.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full_like(x, fill_value)


def segment_op(data, segment_ids, num_out, opname):
    """segmet_max, segmet_min, segmet_mean, segmet_sum, segmet_prod
       unsorted_segment_max, unsorted_segmet_min, unsorted_segmet_mean,
       unsorted_segmet_sum, unsorted_segmet_prod operators.
    Parameters
    ----------
    data : tvm.te.Tensor
        input data
    segment_ids : tvm.te.Tensor
        input segment ids
    num_out : int
        number of output
    opname : str
        name of target op
    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape determined by the segment ids.
    """

    def _max(in_data, out):
        return te.max(in_data, out)

    def _min(in_data, out):
        return te.min(in_data, out)

    def _add(in_data, out):
        return in_data + out

    def _prod(in_data, out):
        return in_data * out

    func_dict = {"max": _max, "min": _min, "mean": _add, "sum": _add, "prod": _prod}
    init_dict = {
        "max": -float("inf"),
        "min": float("inf"),
        "mean": 0.0,
        "sum": 0.0,
        "prod": 1.0,
    }

    def _segment_op(data, segment_ids, out_buf):
        func = func_dict[opname]
        init_data = init_dict[opname]

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        seg_ids = ib.buffer_ptr(segment_ids)
        out = ib.buffer_ptr(out_buf)
        num = ib.allocate("int32", (1), name="num", scope="local")

        shape = get_const_tuple(data.shape)
        num_segment = get_const_tuple(out_buf.shape)[0]
        # The number of data to be calculated for each output
        inner_size = 1
        for s in range(1, len(shape)):
            inner_size = inner_size * shape[s]

        # Init output tensor
        with ib.for_range(0, num_segment) as n:
            with ib.for_range(0, inner_size) as j:
                out_index = n * inner_size + j
                out[out_index] = init_data

            num[0] = 0
            # Operate on numbers with the same id
            with ib.for_range(0, shape[0]) as k:
                with ib.if_scope(seg_ids[k] == n):
                    with ib.for_range(0, inner_size) as l:
                        out_index = n * inner_size + l
                        in_index = k * inner_size + l
                        out[out_index] = func(input_data[in_index], out[out_index])
                    num[0] += 1
            if opname == "mean":
                with ib.if_scope(num[0] > 0):
                    with ib.for_range(0, inner_size) as s:
                        out_index = n * inner_size + s
                        out[out_index] = out[out_index] / num[0]

        return ib.get()

    assert len(segment_ids.shape) == 1

    out_shape = list(get_const_tuple(data.shape))
    out_shape[0] = num_out

    out = te.extern(
        out_shape,
        [data, segment_ids],
        lambda ins, outs: _segment_op(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
    )

    return out


def cumsum(data, axis, exclusive):
    """Cunsum operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    axis : int
        The axis to computed.

    exclusive : bool
        If True, perform exclusive cumsum or cumpros.

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape same as input data.
    """

    def _cumsum(data, out_buf, axis, exclusive):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        out = ib.buffer_ptr(out_buf)
        temp = ib.allocate("float32", (1,), name="temp_data", scope="local")

        shape = get_const_tuple(data.shape)
        outer_size = 1
        for i, s in enumerate(shape):
            if i < axis:
                outer_size = outer_size * s

        inner_size = 1
        for i, s in enumerate(shape):
            if i > axis:
                inner_size = inner_size * s

        cnt = shape[axis]

        with ib.for_range(0, outer_size) as i:
            with ib.for_range(0, inner_size) as k:
                temp[0] = 0.0
                with ib.for_range(0, cnt) as j:
                    index = i * inner_size * cnt + j * inner_size + k
                    temp[0] = temp[0] + input_data[index]
                    if not exclusive:
                        out[index] = temp[0]
                    else:
                        out[index] = temp[0] - input_data[index]

        return ib.get()

    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: _cumsum(ins[0], outs[0], axis, exclusive),
        dtype=data.dtype,
    )

    return out


def cumprod(data, axis, exclusive):
    """Cunprod operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    axis : int
        The axis to computed.

    exclusive : bool
        If True, perform exclusive cumprod or cumpros.

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape same as input data.
    """

    def _cumprod(data, out_buf, axis, exclusive):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        out = ib.buffer_ptr(out_buf)
        temp = ib.allocate("float32", (1,), name="temp_data", scope="local")

        shape = get_const_tuple(data.shape)
        outer_size = 1
        for i, s in enumerate(shape):
            if i < axis:
                outer_size = outer_size * s

        inner_size = 1
        for i, s in enumerate(shape):
            if i > axis:
                inner_size = inner_size * s

        cnt = shape[axis]

        with ib.for_range(0, outer_size) as i:
            with ib.for_range(0, inner_size) as k:
                temp[0] = 1.0
                with ib.for_range(0, cnt) as j:
                    index = i * inner_size * cnt + j * inner_size + k
                    temp[0] = temp[0] * input_data[index]
                    if not exclusive:
                        out[index] = temp[0]
                    else:
                        out[index] = temp[0] / input_data[index]

        return ib.get()

    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: _cumprod(ins[0], outs[0], axis, exclusive),
        dtype=data.dtype,
    )

    return out


def hardmax(data, axis):
    """hardmax operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape same as input data.
    """

    def _hardmax(data, out_buf, axis):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        out = ib.buffer_ptr(out_buf)
        temp = ib.allocate("float32", (1,), name="temp_data", scope="local")
        temp1 = ib.allocate("int32", (1,), name="temp1_data", scope="local")
        shape = get_const_tuple(data.shape)
        if axis.value == -1:
            axis = len(data.shape) - 1
        else:
            axis = axis.value
        out_size = prod(data.shape)

        with ib.for_range(0, out_size) as i:
            out[i] = 0.0

        outer_size = 1
        for i, s in enumerate(shape):
            if i < axis:
                outer_size = outer_size * s

        inner_size = 1
        for i, s in enumerate(shape):
            if i > axis:
                inner_size = inner_size * s

        cnt = shape[axis]
        with ib.for_range(0, outer_size, "i0") as i:
            with ib.for_range(0, inner_size, "k") as k:
                temp[0] = -float("inf")
                temp1[0] = i * inner_size * cnt + k
                with ib.for_range(0, cnt, "j") as j:
                    index = i * inner_size * cnt + j * inner_size + k
                    with ib.if_scope(input_data[index] > temp[0]):
                        temp1[0] = index
                        temp[0] = input_data[index]

                out[temp1[0]] = 1.0

        return ib.get()

    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: _hardmax(ins[0], outs[0], axis),
        dtype=data.dtype,
    )

    return out


def celu(data, alpha):
    """celu operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape same as input data.
    """

    def _celu(data, out_buf, alpha):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        out = ib.buffer_ptr(out_buf)
        out_size = prod(data.shape)

        with ib.for_range(0, out_size) as i:
            out[i] = te.max(0, input_data[i]) + te.min(
                0, alpha * (te.exp(input_data[i] / alpha) - 1)
            )

        return ib.get()

    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: _celu(ins[0], outs[0], alpha),
        dtype=data.dtype,
    )

    return out
