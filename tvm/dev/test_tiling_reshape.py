import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestCase():

    def __init__(self, test_name, input_shape, dtype, sram_size):
        self.name = test_name
        self.input_shape = input_shape
        self.dtype = dtype
        self.sram_size = sram_size

        # setup data
        self.np_data = np.random.rand(*self.input_shape).astype(self.dtype)
        self.tvm_data = tvm.nd.array(self.np_data)

        # quantization parameters
        self.QConfig = relay.quantize.qconfig(
            network_name = self.name,
            have_prequantized = False,
            estimator_activation = "min_max",
            estimator_weight = "min_max",
            quantizer_weight = "Symmetric",
            quantizer_activation = "Symmetric",
            nbit_input = 8,
            nbit_weight = 8,
            per_channel = True,
            do_simulation = False,
            debug_mode = True, # print calibration's consine similarity 
            calibrate_chunk_by = 16,
            cache_size = 800,
        )


    def run(self, target="llvm", params={}):
        logging.info(f"++++++++++++++++++++++ TEST: {self.name} ++++++++++++++++++++++")
        # convert network
        self.net = self.network()
        self.mod=relay.Function(relay.analysis.free_vars(self.net), self.net)
        self.mod_original = tvm.IRModule.from_expr(self.mod)
        logging.info(f"SRAM size = {self.sram_size} Bytes")
        logging.info("--------- Original Relay ---------")
        logging.info(self.mod_original)

        mod_tiling = relay.transform.LayerGroupTiling(self.mod_original, self.sram_size, 4)
        optimizepass = tvm.transform.Sequential([tvm.relay.transform.InferType()])
        mod_tiling = optimizepass(mod_tiling)

        logging.info("--------- After Tiling ---------")
        logging.info(mod_tiling)

        with tvm.transform.PassContext(opt_level=3):
            with self.QConfig:
                quantized_lib = relay.build(self.mod_original, target=target, params=params)

        # run quantized model
        dev = tvm.device(str(target), 0)
        compiled_module = graph_executor.GraphModule(quantized_lib["default"](dev))
        compiled_module.set_input("data", self.tvm_data)
        compiled_module.run()
        network_output = compiled_module.get_output(0).numpy()
        logging.info(f"output shape: {network_output.shape}")

        # compare the original result and tiling result
        np.testing.assert_allclose(network_output, self.target_infer(self.np_data), rtol=1e-3, atol=1e-5)
        logging.info("PASS")

        return True

    def target_infer(self):
        raise NotImplementedError("Not Implemented")
    
    def network(self):
        raise NotImplementedError("Not Implemented")
    

class Test_1(TestCase):
    def __init__(self, test_name, input_shape, sram_size, output_shape, dtype="float32"):
        super().__init__(test_name, input_shape, dtype, sram_size)
        self.output_shape = output_shape

    def target_infer(self, np_data):
        return np_data.reshape(self.output_shape)
    
    def network(self):
        data = relay.var("data", shape=self.input_shape, dtype=self.dtype)
        op = relay.reshape(
            data=data,
            newshape=self.output_shape,
        )
        return op

class Test_2(TestCase):
    def __init__(self, test_name, input_shape, sram_size, dtype="float32", **kwargs):
        super().__init__(test_name, input_shape, dtype, sram_size)
        self.mid_shape = kwargs["mid_shape"]
        self.output_shape = kwargs["output_shape"]
    
    def target_infer(self, np_data):
        return np_data.reshape(self.output_shape)

    def network(self):
        data = relay.var("data", shape=self.input_shape, dtype=self.dtype)
        op = relay.reshape(
            data=data,
            newshape=self.mid_shape,
        )
        op = relay.reshape(
            data=op,
            newshape=self.output_shape
        )
        return op

class Test_3(TestCase):
    def __init__(self, test_name, input_shape, sram_size, dtype="float32", **kwargs):
        super().__init__(test_name, input_shape, dtype, sram_size)
        self.output_shape = kwargs["output_shape"]
        
    def target_infer(self, np_data: np.ndarray):
        torch_data = torch.from_numpy(np_data)
        weight = torch.ones((64, 32, 3, 3))
        bias = torch.ones((64,))
        torch_out = F.relu(torch_data)
        torch_out = F.conv2d(
            input=torch_out,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            dilation=1)
        torch_out = torch_out.view(self.output_shape)
        return torch_out.detach().numpy()

    def network(self):
        kernel_size = (3, 3)
        strides = (1, 1)
        padding = (0, 0, 0, 0)
        channels = 64
        dilation = (1, 1)
        weight_shape = (64, 32, 3, 3)
        bias_shape = (64,)

        data = relay.var("data", shape=self.input_shape, dtype=self.dtype)
        np_weight = np.ones(weight_shape).astype(self.dtype)
        weight = relay.const(np_weight)
        np_bias = np.ones(bias_shape).astype(self.dtype)
        bias = relay.const(np_bias)

        op = relay.nn.relu(data=data)
        op = relay.nn.conv2d(data=op,
                             weight=weight,
                             channels=channels,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             data_layout="NCHW")
        op = relay.nn.bias_add(op, bias)
        op = relay.reshape(data=op, newshape=self.output_shape)
        return op


if __name__ == "__main__":

    op_name = "reshape_unit_test"

    # logging setup
    savepath = "./" + op_name
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    savepath = savepath + "/"
    logpath = savepath + op_name + time.strftime('_%y-%m-%d', time.localtime()) + ".log"
    logging.basicConfig(level=logging.DEBUG,
                        filename=logpath,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode="a")
    
    # initialize test cases
    test_1 = Test_1(test_name="single reshape",
                    input_shape=(1, 50, 24, 24),
                    sram_size=32*1024,
                    output_shape=(1, 2, 25, 24, 24))

    test_2 = Test_2(test_name="two reshapes",
                    input_shape=(1, 50, 24, 2, 12),
                    sram_size=32*1024,
                    mid_shape=(1, 50, 24, 24),
                    output_shape=(1, 2, 25, 24, 24))

    test_3 = Test_3(test_name="relu-conv-reshape",
                    input_shape=(1, 32, 14, 14),
                    sram_size=32*1024,
                    output_shape=(1, 64, 12*12))


    test_case_list = [
        test_1,
        # test_2,
        # test_3,
    ]

    for idx, test_case in enumerate(test_case_list):
        try:
            success = False
            success = test_case.run()
            if success:
                print(f"Test{idx+1} - {test_case.name:<20} ................. {bcolors.OKGREEN}PASS{bcolors.ENDC}")
        except Exception as e:
            logging.error(e)
            print(f"Test{idx+1} - {test_case.name:<20} ................. {bcolors.FAIL}FAIL{bcolors.ENDC}")
        finally:
            logging.info("-"*80 + "\n\n")
