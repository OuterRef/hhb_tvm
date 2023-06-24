/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* SHL version 2.1.x */

#include "csi_nn.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of floor div f32.\n");

    struct csinn_tensor *input0 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *input1 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_diso_params *params = csinn_alloc_params(sizeof(struct csinn_diso_params), NULL);
    int in_size = 0;
    int out_size = 0;

    int *buffer = read_input_data_f32(argv[1]);

    input0->dim[0] = input1->dim[0] = buffer[0];  // batch
    input0->dim[1] = input1->dim[1] = buffer[1];  // channel
    input0->dim[2] = input1->dim[2] = buffer[2];  // height
    input0->dim[3] = input1->dim[3] = buffer[3];  // width

    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];

    in_size = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    out_size = in_size;
    input0->dim_count = 4;
    input1->dim_count = 4;
    output->dim_count = 4;
    input0->dtype = CSINN_DTYPE_FLOAT32;
    input1->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params->base.api = CSINN_API;

    input0->data = (float *)(buffer + 4);
    input1->data = (float *)(buffer + 4 + in_size);
    reference->data = (float *)(buffer + 4 + 2 * in_size);
    output->data = (float *)malloc(out_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_floor_divide_init(input0, input1, output, params) == CSINN_TRUE) {
        csinn_floor_divide(input0, input1, output, params);
    }

    result_verify_f32(reference->data, output->data, input0->data, difference, in_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
