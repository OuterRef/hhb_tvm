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
# pylint: disable=invalid-name
"""Common riscv related utilities"""
import tvm
from tvm import te


@tvm._ffi.register_func("tvm.topi.riscv.utils.target_has_sse41")
def target_has_sse41(target):
    return (
        target_has_sse42(target)
        or target_has_avx(target)
        or target_has_avx2(target)
        or target_has_avx512(target)
        or target_has_vnni(target)
        or target
        in {
            "btver2",
            "penryn",
        }
    )


@tvm._ffi.register_func("tvm.topi.riscv.utils.target_has_sse42")
def target_has_sse42(target):
    return (
        target_has_avx(target)
        or target_has_avx2(target)
        or target_has_avx512(target)
        or target_has_vnni(target)
        or target
        in {
            "silvermont",
            "slm",
            "goldmont",
            "goldmont-plus",
            "tremont",
            "nehalem",
            "corei7",
            "westmere",
            "bdver1",
            "bdver2",
            "bdver3",
            "riscv-64-v2",
        }
    )


@tvm._ffi.register_func("tvm.topi.riscv.utils.target_has_avx")
def target_has_avx(target):
    return (
        target_has_avx2(target)
        or target_has_avx512(target)
        or target_has_vnni(target)
        or target in {"sandybridge", "corei7-avx", "ivybridge", "core-avx-i"}
    )


@tvm._ffi.register_func("tvm.topi.riscv.utils.target_has_avx2")
def target_has_avx2(target):
    return (
        target_has_avx512(target)
        or target_has_vnni(target)
        or target
        in {
            "haswell",
            "core-avx2",
            "broadwell",
            "skylake",
            "bdver4",
            "znver1",
            "znver2",
            "znver3",
            "riscv-64-v3",
        }
    )


@tvm._ffi.register_func("tvm.topi.riscv.utils.target_has_avx512")
def target_has_avx512(target):
    return target in {
        "skylake-avx512",
        "skx",
        "knl",
        "knm",
        "riscv-64-v4",
        "cannonlake",
        # explicit enumeration of VNNI capable due to collision with alderlake
        "cascadelake",
        "icelake-client",
        "icelake-server",
        "rocketlake",
        "tigerlake",
        "cooperlake",
        "sapphirerapids",
    }


@tvm._ffi.register_func("tvm.topi.riscv.utils.target_has_vnni")
def target_has_vnni(target):
    return target in {
        "cascadelake",
        "icelake-client",
        "icelake-server",
        "rocketlake",
        "tigerlake",
        "cooperlake",
        "sapphirerapids",
        "alderlake",
    }


@tvm._ffi.register_func("tvm.topi.riscv.utils.get_simd_32bit_lanes")
def get_simd_32bit_lanes():
    "riscv vector length"
    cfg = tvm.target.Target.current()
    keys = cfg.keys
    fp32_vec_len = 4

    for key in keys:
        if "m" in key:
            if key == "m2":
                fp32_vec_len *= 2
                break
            if key != "m1":
                raise Exception("only support m1/m2 now.")

    return fp32_vec_len


tvm.target.datatype.register("vfloat32m1_t", 129)
tvm.target.datatype.register_op(
    tvm.target.datatype.lower_call_pure_extern,
    "Call",
    "c",
    "vfloat32m1_t",
    intrinsic_name="tir.call_extern",
)

tvm.target.datatype.register("vfloat32m2_t", 130)
tvm.target.datatype.register_op(
    tvm.target.datatype.lower_call_pure_extern,
    "Call",
    "c",
    "vfloat32m2_t",
    intrinsic_name="tir.call_extern",
)

tvm.target.datatype.register("vfloat16m1_t", 131)
tvm.target.datatype.register_op(
    tvm.target.datatype.lower_call_pure_extern,
    "Call",
    "c",
    "vfloat16m1_t",
    intrinsic_name="tir.call_extern",
)

tvm.target.datatype.register("vfloat16m2_t", 132)
tvm.target.datatype.register_op(
    tvm.target.datatype.lower_call_pure_extern,
    "Call",
    "c",
    "vfloat16m2_t",
    intrinsic_name="tir.call_extern",
)


def rv_intrin_dtype(dtype):
    if dtype not in ["float32", "float16"]:
        assert 0
    out = "f32" if dtype == "float32" else "f16"
    return out


def get_vse_intrin(dtype, m):
    rv_dtype = rv_intrin_dtype(dtype)

    return f"vse{dtype[-2:]}_v_{rv_dtype}m{m}"


def get_vle_intrin(dtype, m):
    rv_dtype = rv_intrin_dtype(dtype)

    return f"vle{dtype[-2:]}_v_{rv_dtype}m{m}"


def load_vec(vec, l, ib, dtype, name="vec"):
    vl = get_simd_32bit_lanes()
    m = vl // 4
    vle_intrin = get_vle_intrin(dtype, m)
    out = tvm.tir.call_extern(f"custom[v{dtype}m{m}_t]{vl * 32}", vle_intrin, vec, l)

    return ib.let(name, out)


def intrin_macc_vf(l, dtype, set_to_out=False):
    """match math sum(a[k] * b[0], axis=k)"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    b = te.placeholder((1,), name="b", dtype=dtype)
    k = te.reduce_axis((0, l), name="k")
    c = te.compute((1,), lambda i: te.sum(a[k] * b[0], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("s1")])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    rv_dtype = rv_intrin_dtype(dtype)
    vl = get_simd_32bit_lanes()
    m = vl // 4

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        vec_c = cc.vload([0], dtype)

        def _body():
            ib = tvm.tir.ir_builder.create()
            a = aa.vload([0], dtype)
            vec_b = bb.access_ptr("r")

            # d = a * b + c
            # load vec_b
            load_vecb = load_vec(vec_b, l, ib, dtype)
            load_vecb = ib.let("vec", load_vecb)

            # do macc
            vmmla = tvm.tir.call_pure_extern(
                dtype, f"vfmacc_vf_{rv_dtype}m{m}", vec_c, a, load_vecb, l
            )
            ib.emit(cc.vstore([0], vmmla))
            if set_to_out:
                vse_intrin = get_vse_intrin(dtype, m)
                ib.emit(tvm.tir.call_extern(dtype, vse_intrin, cc.access_ptr("r"), vec_c, l))

            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()

            init_value = te.const(0, dtype)
            ib.emit(tvm.tir.call_extern(dtype, f"vfmv_v_f_{rv_dtype}m{m}", init_value, l, vec_c))

            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def get_add_act(inputs, shape, act_mod):
    "act compute pattern"
    a = inputs[0]
    b = inputs[1]
    dtype = a.dtype
    if act_mod == "relu":
        min_ = te.const(0, dtype)
        return te.compute(shape, lambda i: te.max(a[i] + b[i], min_), name="add_relu")
    elif act_mod == "relu6":
        min_ = te.const(0, dtype)
        max_ = te.const(6, dtype)
        return te.compute(
            shape, lambda i: te.max(te.min(a[i] + b[i], max_), min_), name="add_relu6"
        )
    else:
        raise Exception(f"activation function '{act_mod}' not register!")


def get_add_add_act(inputs, shape, act_mod, post=False):
    "act compute pattern"
    a = inputs[0]
    b = inputs[1]
    c = inputs[2]
    dtype = a.dtype
    if act_mod == "relu":
        min_ = te.const(0, dtype)
        out = te.compute(shape, lambda i: te.max((a[i] + b[i]) + c[i], min_), name="add_relu")
        if post:
            out = te.compute(shape, lambda i: te.max(a[i] + (b[i] + c[i]), min_), name="add_relu")
    elif act_mod == "relu6":
        min_ = te.const(0, dtype)
        max_ = te.const(6, dtype)
        out = te.compute(
            shape, lambda i: te.max(te.min((a[i] + b[i]) + c[i], max_), min_), name="add_relu6"
        )
        if post:
            out = te.compute(
                shape, lambda i: te.max(te.min(a[i] + (b[i] + c[i]), max_), min_), name="add_relu6"
            )
    else:
        raise Exception(f"activation function '{act_mod}' not register!")
    return out


def get_act_intrin(act_mod, ib, in_data, l, dtype):
    "act compute intrinsic"
    vl = get_simd_32bit_lanes()
    m = vl // 4

    rv_dtype = rv_intrin_dtype(dtype)
    intrin_dtype = f"{rv_dtype}m{m}"
    if act_mod == "relu":
        zero = te.const(0, dtype)
        relu = tvm.tir.call_extern(
            f"custom[v{dtype}m{m}_t]{vl * 32}", f"vfmax_vf_{intrin_dtype}", in_data, zero, l
        )
        out = ib.let("relu_vf", relu)
    elif act_mod == "relu6":
        min_ = te.const(0, dtype)
        max_ = te.const(6, dtype)
        relu = tvm.tir.call_extern(
            f"custom[v{dtype}m{m}_t]{vl * 32}", f"vfmax_vf_{intrin_dtype}", in_data, min_, l
        )
        relu = ib.let("relu_vf", relu)
        relu6 = tvm.tir.call_extern(
            f"custom[v{dtype}m{m}_t]{vl * 32}", f"vfmin_vf_{intrin_dtype}", relu, max_, l
        )
        out = ib.let("relu6_vf", relu6)
    else:
        raise Exception(f"activation function '{act_mod}' not register!")

    return out


def intrin_act(l, act_mod, dtype, load=False):
    """match math a[i] + b[i]"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    shape = a.shape

    if act_mod == "relu":
        min_ = te.const(0, dtype)
        c = te.compute(shape, lambda i: te.max(a[i], min_), name="add_relu")
    elif act_mod == "relu6":
        min_ = te.const(0, dtype)
        max_ = te.const(6, dtype)
        c = te.compute(shape, lambda i: te.max(te.min(a[i], max_), min_), name="add_relu6")
    else:
        raise Exception(f"activation function '{act_mod}' not register!")

    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    vl = get_simd_32bit_lanes()
    m = vl // 4

    def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]

        ib = tvm.tir.ir_builder.create()

        vec_a = aa.vload([0], dtype)
        # load input
        if load:
            vec_a = aa.access_ptr("r")
            vec_a = load_vec(vec_a, l, ib, dtype, "va")
        vec_c = cc.access_ptr("w")

        out = get_act_intrin(act_mod, ib, vec_a, l, dtype)
        vse_intrin = get_vse_intrin(dtype, m)
        ib.emit(tvm.tir.call_extern(dtype, vse_intrin, vec_c, out, l))

        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb})


def intrin_add_vv(l, dtype, load_a=True, load_b=True, act_mod=None):
    """match math a[i] + b[i]"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    b = te.placeholder((l,), name="b", dtype=dtype)

    if act_mod:
        c = get_add_act((a, b), a.shape, act_mod)
    else:
        c = te.compute(a.shape, lambda i: a[i] + b[i], name="add")

    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, a.dtype, name="B", offset_factor=1, strides=[1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    vl = get_simd_32bit_lanes()
    m = vl // 4

    rv_dtype = rv_intrin_dtype(dtype)

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        ib = tvm.tir.ir_builder.create()

        vec_a = aa.vload([0], dtype)
        vec_b = bb.access_ptr("r")
        vec_c = cc.access_ptr("w")

        # load a
        if load_a:
            vec_a = load_vec(vec_a, l, ib, dtype, "va")
        # load b
        if load_b:
            vec_b = load_vec(vec_b, l, ib, dtype, "vb")

        vadd = tvm.tir.call_extern(
            f"custom[v{dtype}m{m}_t]{vl * 32}", f"vfadd_vv_{rv_dtype}m{m}", vec_a, vec_b, l
        )
        out = ib.let("add_vv", vadd)
        if act_mod:
            out = get_act_intrin(act_mod, ib, out, l, dtype)
        vse_intrin = get_vse_intrin(dtype, m)
        ib.emit(tvm.tir.call_extern(dtype, vse_intrin, vec_c, out, l))

        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def intrin_load(l, dtype):
    """match math load a[i]"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    c = te.compute((l,), lambda i: a[i], name="load")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    vl = get_simd_32bit_lanes()
    m = vl // 4

    def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]

        ib = tvm.tir.ir_builder.create()

        vec_a = aa.vload([0], dtype)
        vec_c = cc.access_ptr("r")
        load_intrin = get_vse_intrin(dtype, m)
        ib.emit(tvm.tir.call_extern(dtype, load_intrin, vec_c, vec_a, l))

        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb})


def intrin_macc_fv(l, dtype):
    """match math sum(a[0] * b[i], axis=k)"""
    a = te.placeholder((1,), name="a", dtype=dtype)
    b = te.placeholder((l,), name="b", dtype=dtype)
    k = te.reduce_axis((0, l), name="k")
    c = te.compute((1,), lambda i: te.sum(a[0] * b[i], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("s1")])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    rv_dtype = rv_intrin_dtype(dtype)
    vl = get_simd_32bit_lanes()
    m = vl // 4

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        vec_c = cc.vload([0], dtype)

        def _body():
            ib = tvm.tir.ir_builder.create()
            vec_a = aa.vload([0], dtype)
            vec_b = bb.access_ptr("r")

            # d = a * b + c
            load_vecb = load_vec(vec_b, l, ib, dtype)

            # do macc
            vmmla = tvm.tir.call_pure_extern(
                dtype, f"vfmacc_vf_{rv_dtype}m{m}", vec_c, vec_a, load_vecb, l
            )
            ib.emit(cc.vstore([0], vmmla))

            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            init_value = te.const(0, dtype)
            ib.emit(tvm.tir.call_extern(dtype, f"vfmv_v_f_{rv_dtype}m{m}", init_value, l, vec_c))

            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def intrin_macc_vv(l, dtype):
    """match math sum(a[i] * b[i], axis=k)"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    b = te.placeholder((l,), name="b", dtype=dtype)
    k = te.reduce_axis((0, l), name="k")
    c = te.compute((l,), lambda i: te.sum(a[i] * b[i], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[te.var("s1")])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[te.var("s1")])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    rv_dtype = rv_intrin_dtype(dtype)
    vl = get_simd_32bit_lanes()
    m = vl // 4

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        vec_c = cc.vload([0], dtype)

        def _body():
            ib = tvm.tir.ir_builder.create()
            vec_a = aa.access_ptr("r")
            vec_b = bb.access_ptr("r")

            # d = a * b + c
            load_veca = load_vec(vec_a, l, ib, dtype)
            load_vecb = load_vec(vec_b, l, ib, dtype)

            # do macc
            vmmla = tvm.tir.call_pure_extern(
                dtype, f"vfmacc_vv_{rv_dtype}m{m}", vec_c, load_veca, load_vecb, l
            )
            ib.emit(cc.vstore([0], vmmla))

            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            init_value = te.const(0, dtype)
            ib.emit(tvm.tir.call_extern(dtype, f"vfmv_v_f_{rv_dtype}m{m}", init_value, l, vec_c))

            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def is_relu6(compute):
    c_str = str(compute)
    if "max" in c_str and "min" in c_str:
        op1 = c_str[:3] == "max"
        op2 = c_str[4:7] == "min"
        vmax = compute.a.b.value == 6
        vmin = compute.b.value == 0
        return op1 and op2 and vmax and vmin

    return False


def get_act_mod(c_op):
    if is_relu6(c_op.body[0]):
        act_mod = "relu6"
    else:
        act_mod = c_op.name[2:]

    return act_mod


def intrin_mul_vf(l, f_value, dtype, load_a=False):
    """match avg pool"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    b = te.const(f_value, dtype)

    c = te.compute(a.shape, lambda i: a[i] * b, name="add")

    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[te.var("s1")])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    rv_dtype = rv_intrin_dtype(dtype)
    vl = get_simd_32bit_lanes()
    m = vl // 4

    def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]

        ib = tvm.tir.ir_builder.create()

        vec_a = aa.vload([0], dtype)
        if load_a:
            vec_a = load_vec(aa.access_ptr("r"), l, ib, dtype, "veca")
        vec_c = cc.access_ptr("w")

        vadd = tvm.tir.call_extern(
            f"custom[v{dtype}m{m}_t]{vl * 32}", f"vfmul_vf_{rv_dtype}m{m}", vec_a, b, l
        )
        out = ib.let("add_vf", vadd)

        vse_intrin = get_vse_intrin(dtype, m)
        ib.emit(tvm.tir.call_extern(dtype, vse_intrin, vec_c, out, l))

        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb})


def intrin_add_add_vv(l, dtype, load_a=True, load_b=True, load_c=True, act_mod=None, post=False):
    """match math a[i] + b[i]"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    b = te.placeholder((l,), name="b", dtype=dtype)
    c = te.placeholder((l,), name="c", dtype=dtype)

    if act_mod:
        d = get_add_add_act((a, b, c), a.shape, act_mod, post)
    else:
        d = te.compute(a.shape, lambda i: a[i] + b[i] + c[i], name="add")
        if post:
            d = te.compute(a.shape, lambda i: a[i] + (b[i] + c[i]), name="add")

    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, a.dtype, name="B", offset_factor=1, strides=[1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])
    Db = tvm.tir.decl_buffer(c.shape, c.dtype, name="D", offset_factor=1, strides=[1])

    vl = get_simd_32bit_lanes()
    rv_dtype = rv_intrin_dtype(dtype)
    m = vl // 4

    def intrin_func(ins, outs):
        aa, bb, cc = ins
        dd = outs[0]

        ib = tvm.tir.ir_builder.create()

        vec_a = aa.vload([0], dtype)
        if load_a:
            vec_a = aa.access_ptr("r")
            vec_a = load_vec(vec_a, l, ib, dtype, "va")

        vec_b = bb.vload([0], dtype)
        if load_b:
            vec_b = bb.access_ptr("r")
            vec_b = load_vec(vec_b, l, ib, dtype, "vb")

        vec_c = cc.vload([0], dtype)
        if load_c:
            vec_c = cc.access_ptr("r")
            vec_c = load_vec(vec_c, l, ib, dtype, "vc")

        vec_d = dd.access_ptr("w")

        # load a

        vadd1 = tvm.tir.call_extern(
            f"custom[v{dtype}m{m}_t]{vl * 32}", f"vfadd_vv_{rv_dtype}m{m}", vec_b, vec_c, l
        )
        out1 = ib.let("add_vv", vadd1)
        vadd2 = tvm.tir.call_extern(
            f"custom[v{dtype}m{m}_t]{vl * 32}", f"vfadd_vv_{rv_dtype}m{m}", vec_a, out1, l
        )
        out = ib.let("add_vv", vadd2)

        if act_mod:
            out = get_act_intrin(act_mod, ib, out, l, dtype)

        vse_intrin = get_vse_intrin(dtype, m)
        ib.emit(tvm.tir.call_extern(dtype, vse_intrin, vec_d, out, l))

        return ib.get()

    return te.decl_tensor_intrin(d.op, intrin_func, binds={a: Ab, b: Bb, c: Cb, d: Db})
