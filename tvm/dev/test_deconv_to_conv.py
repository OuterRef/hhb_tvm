import tvm
import tvm.relay as relay
import numpy as np
import os
import sys
from tvm.contrib import graph_executor

# from tvm.relay.transform.transform import *

#####################################
# Read Network (the section need change)
#####################################

op_name = "conv_unit_test" # the name of save_dir and txt
dtype = "float32"   # the Tensor data type
data_shape = (1, 32, 256, 256)  # the Tensor data shape
weight_shape = (16, 32, 2, 2)  # the Tensor weight shape
# bias_shape = (64,)
have_weight = True
have_bias = True
per_channel = True

# the calibrate key name
op_func_name = "nn.conv2d"
input_key = "network_input_0" # the input key name in act_calibrate.json
output_key = op_func_name + "_0:out" # the output key name in act_calibrate.json
weight_key = op_func_name + "_weight_0_0:in" # the weight key name in wgt_calibrate.json

# the Qconfig params
quantizer_weight = "Symmetric" # the quantize type of weight
quantizer_activation = "Symmetric" # the quantize type of input and output
estimator_activation = "min_max"
nbit_input = 8
nbit_weight = 8
do_simulation = False

#setup data
np_data = tvm.nd.array(np.random.rand(1, 32, 256, 256).astype(dtype))
# np.random.uniform(-1, 1, data_shape)
# np_data = np.ones((1, 3, 224, 224))
# np_weight = tvm.nd.array(np.random.uniform(-1, 1, weight_shape).astype(dtype))

print("input")
print(np_data)

# np_weight = np.ones(weight_shape).astype(dtype)
np_weight = tvm.nd.array(np.random.rand(16, 32, 2, 2).astype(dtype))
print("weight")
print(np_weight)
params = {
        "weight": np_weight,
}


#create the save_dir
# savepath = "./" + op_name
# if not os.path.exists(savepath):
#     os.mkdir(savepath)
# savepath = savepath + "/"

# create the op net
# def simplenet():    

#     kernel_size = (3, 3)
#     strides = (1, 1)
#     padding = (0, 0)
#     channels = 1
#     dilation = 1

#     data = relay.var("data", shape=data_shape, dtype=dtype)
#     weight = relay.var("weight",dtype=dtype)


#     op = relay.nn.conv2d(
#             data=data,
#             weight=weight,
#             channels=channels,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_layout='NCHW')

#     # save the infp to txt
#     return op
etab = relay.frontend.common.ExprTable()

def simplenet():    

    kernel_size = (2,2)
    strides = (2,2)
    padding = (0,0,0,0)
    channels = weight_shape[0]

    data = relay.var("data", shape=data_shape, dtype=dtype)
#     weight = etab.new_const(np_weight)
    weight = relay.var("weight",dtype=dtype)
    op = relay.nn.conv2d_transpose(
            data=data,
            weight=weight,
            channels=channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_layout='OIHW',
            data_layout='NCHW')

    return op
#####################################
# Determine quantization parameters
#####################################

QConfig = relay.quantize.qconfig(
        network_name = op_name,
        have_prequantized = False,
        estimator_activation = estimator_activation,
        estimator_weight = "min_max",
        quantizer_weight = quantizer_weight,
        quantizer_activation = quantizer_activation,
        nbit_input = nbit_input,
        nbit_weight = nbit_weight,
        per_channel = per_channel,
        do_simulation = do_simulation,
        debug_mode = True, # print calibration's consine similarity 
        calibrate_chunk_by = 16,
        cache_size = 800,
        )

#####################################
# Convert Network
#####################################

#create model from relay
net=simplenet()
mod=relay.Function(relay.analysis.free_vars(net), net)
mod_original = tvm.IRModule.from_expr(mod)

#print(mod_original)
if params:
    mod_original["main"] = tvm.relay.build_module.bind_params_by_name(mod_original["main"], params)

seq = tvm.transform.Sequential(
    [
    relay.transform.InferType(),
    ]
)
seq(mod_original)
target = "llvm"

#####################################
# Prepare calibration dataset
#####################################
data_set=[]
for x in range(1):
    data_set.append({"data":np_data})

dev = tvm.device(str(target), 0)

print("ori\n")
print(mod_original)

# with tvm.transform.PassContext(opt_level=3):
#     with QConfig:
#         mod_tiling, dbg_mod = relay.transform.LayerGroupTiling(mod_original, params=params)
# mod_tiling = relay.transform.LayerGroupTiling(mod_original)

# print("tiling\n")
# print(mod_tiling)
# mod_original = relay.transform.Deconv2dToConv2d(mod_original)

# print("transpose\n")
# print(mod_original)
######################################
# Build and Run the quantized model
######################################

with tvm.transform.PassContext(opt_level=3):
    with QConfig:
        quantized_lib = relay.build(mod_original, target=target)

compiled_module = graph_executor.GraphModule(quantized_lib["default"](dev))
compiled_module.set_input("data",np_data)
compiled_module.run()

## just get the first output
network_output = compiled_module.get_output(0).numpy()
print("ori output")
print(network_output)
print("ori output shape")
print(network_output.shape)


mod_t = relay.transform.Deconv2dToConv2d(mod_original)

print("transpose\n")
print(mod_t)
######################################
# Build and Run the quantized model
######################################

with tvm.transform.PassContext(opt_level=3):
    with QConfig:
        t_lib = relay.build(mod_t, target=target)

compiled_module1 = graph_executor.GraphModule(t_lib["default"](dev))
compiled_module1.set_input("data",np_data)
compiled_module1.run()

## just get the first output
network_output1 = compiled_module1.get_output(0).numpy()
print("network_output1")
print(network_output1)
print("network_output1 shape")
print(network_output1.shape)

if (network_output1.all() == network_output.all()):
    print("allsame")