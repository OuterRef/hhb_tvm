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

#include "shl_gref.h"

int shl_gref_reshape(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reshape_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_RESHAPE, params);
    return CSINN_TRUE;
}

int shl_gref_reshape_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_reshape_params *params)
{
    int index = -1;
    int total_size = csinn_tensor_size(input);
    int reshape_size = 1;
    output->dim_count = params->shape_num;
    for (int i = 0; i < output->dim_count; i++) {
        if (params->shape[i] == -1) {
            index = i;
        } else {
            output->dim[i] = params->shape[i];
            reshape_size *= params->shape[i];
        }
    }
    if (index >= 0) {
        output->dim[index] = total_size / reshape_size;
    }
    return CSINN_TRUE;
}