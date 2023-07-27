import tvm
import tvm.relay as relay
import numpy as np
import os
import sys
from tvm.contrib import graph_executor
import logging
import time

# from tvm.relay.transform.transform import *

#####################################
# Read Network (the section need change)
#####################################

op_name = "conv_unit_test" # the name of save_dir and txt
dtype = "float32"   # the Tensor data type
data_shape = (1, 10, 96, 96)  # the Tensor data shape
weight_shape = (10, 10, 1, 1)  # the Tensor weight shape
bias_shape = (10,)
sram_size = 32*1024
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
np_data = tvm.nd.array(np.random.rand(*data_shape).astype(dtype))
# np.random.uniform(-1, 1, data_shape)
# np_data = np.ones((1, 3, 224, 224))
# np_weight = tvm.nd.array(np.random.uniform(-1, 1, weight_shape).astype(dtype))

# print("input")
# print(np_data)

np_weight = tvm.nd.array(np.ones(weight_shape).astype(dtype))

np_bias = tvm.nd.array(np.ones(bias_shape).astype(dtype))

params = {
}


#create the save_dir
savepath = "./" + op_name
if not os.path.exists(savepath):
    os.mkdir(savepath)
savepath = savepath + "/"
logpath = savepath + op_name + time.strftime('_%y-%m-%d', time.localtime()) + ".log"
logging.basicConfig(level=logging.DEBUG,
                    filename=logpath,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode="a")

# create the op net
def simplenet():    

    kernel_size = (1, 1)
    strides = (1, 1)
    padding = (0, 0, 0, 0)
    channels = 10
    dilation = (1, 1)


    data = relay.var("data", shape=data_shape, dtype=dtype)
    np_weight = np.ones(weight_shape).astype(dtype)
    weight = relay.const(np_weight)
    np_bias = np.ones(bias_shape).astype(dtype)
    bias = relay.const(np_bias)

        
    op = relay.nn.relu(
            data=data,
            )
    op = relay.nn.conv2d(
            data=op,
            weight=weight,
            channels=channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_layout='NCHW')
    
    op = relay.nn.bias_add(op, bias)

    # save the infp to txt
    return op

# def simplenet():    

#     data = relay.var("data", shape=data_shape, dtype=dtype)

#     op0 = relay.nn.relu(
#             data=data,
#             )
#     op1 = relay.nn.relu(
#             data=op0,
#             )
#     op2 = relay.nn.relu(
#             data=op1,
#             )

#     # save the infp to txt
#     return op2
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


target = "llvm"

#####################################
# Prepare calibration dataset
####################################
logging.info(f"SRAM size = {sram_size} Bytes")
logging.info("--------- Original Relay ---------")
logging.info(mod_original)

# with tvm.transform.PassContext(opt_level=3):
#     with QConfig:
#         mod_tiling, dbg_mod = relay.transform.LayerGroupTiling(mod_original, params=params)
# mod_tiling = relay.transform.LayerGroupTiling(mod_original, 200704, 1)
mod_tiling = relay.transform.LayerGroupTiling(mod_original, sram_size, 1)
optimizepass = tvm.transform.Sequential([tvm.relay.transform.InferType()])
mod_tiling = optimizepass(mod_tiling)
logging.info("--------- After Tiling ---------")
logging.info(mod_tiling)
######################################
# Build and Run the quantized model
######################################

with tvm.transform.PassContext(opt_level=3):
    with QConfig:
        quantized_lib = relay.build(mod_original, target=target, params=params)

data_set=[]
for x in range(1):
    data_set.append({"data":np_data})

dev = tvm.device(str(target), 0)
compiled_module = graph_executor.GraphModule(quantized_lib["default"](dev))
compiled_module.set_input("data",np_data)
compiled_module.run()

## just get the first output
network_output = compiled_module.get_output(0).numpy()

# print(network_output)
logging.info(f"output shape: {network_output.shape}")
logging.info("-"*80 + "\n\n")
