/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/qnn/op/quantize.cc
 * \brief QNN quantize operator. Quantize operator converts from unquantized
 * domain to quantized domain.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(QnnCSIQuantizeAttrs);

bool QnnCSIQuantizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) {
    return false;
  }

  const auto input_dtype = data->dtype;
  ICHECK(input_dtype == DataType::Float(32))
      << "Input type should be one of float32 but was " << input_dtype;

  const auto* quantize_attrs = attrs.as<QnnCSIQuantizeAttrs>();
  int axis = quantize_attrs->axis;
  auto rank = static_cast<int>(data->shape.size());
  axis = (axis < 0) ? ((rank > 0) ? data->shape.size() + axis : 0) : axis;

  // If zero point and scale are scalar then axis doesnt matter.
  bool scale_is_scalar, zp_is_scalar;

  if (auto ttype = types[1].as<TensorTypeNode>()) {
    scale_is_scalar = ttype->shape.size() == 0;
  } else {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "Quantize: expect to be TensorType but get " << types[1];
    return false;
  }

  if (auto ttype = types[2].as<TensorTypeNode>()) {
    zp_is_scalar = ttype->shape.size() == 0;
  } else {
    ICHECK(types[2].as<IncompleteTypeNode>())
        << "Quantize: expect to be TensorType but get " << types[2];
    return false;
  }

  if (!(scale_is_scalar && zp_is_scalar)) {
    ICHECK_LT(axis, rank > 0 ? rank : 1) << "axis " << quantize_attrs->axis << " is out of range";
    ICHECK_GE(axis, 0) << "axis " << quantize_attrs->axis << " is out of range";
  }

  PrimExpr axis_shape;
  if (rank > 0) {
    axis_shape = data->shape[axis];
  } else {
    axis_shape = Integer(1);
  }
  // Check and assign types for scale and zero points.
  AssignType(types[1], DataType::Float(32), axis_shape, reporter);  // scale
  AssignType(types[2], DataType::Int(32), axis_shape, reporter);    // zero point

  const Array<tvm::PrimExpr> oshape = data->shape;
  const DataType out_dtype = quantize_attrs->out_dtype;
  ICHECK(out_dtype == DataType::Int(8) || out_dtype == DataType::UInt(8) ||
         out_dtype == DataType::Int(16) || out_dtype == DataType::Int(32))
      << "Output type should be one of [int8, unit8, int16, int32] but was " << out_dtype;
  // assign output type
  reporter->Assign(types[3], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeQnnCSIQuantize(Expr data, Expr output_scale, Expr output_zero_point, int axis,
                        DataType out_dtype, Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSIQuantizeAttrs>();

  attrs->axis = axis;
  attrs->out_dtype = std::move(out_dtype);
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);
  // result_quantized_value = result_zero_point + result_real_value / result_scale.
  // A more detailed explanation can be found here -
  // https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  static const Op& op = Op::Get("qnn.csi.quantize");
  return Call(op, {data, output_scale, output_zero_point}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.quantize")
    .describe(R"code(Quantizes the input and produces quantized output.
The input can be either float or quantized(int8, unit8). If the input is float,
this op takes scale and zero point and quantize the float value to
quantized output, in int8 or uint8 format. If the input is quantized value,
the op requantize the input (of a certain type, with a given scale and zero
point) to the output of the same or different type with a same or different
scale and zero point.
- **data**: Tensor of any shape to quantize. The input data can be of floating point
          or quantized.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIQuantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The tensor to quantize.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIQuantize", QnnCSIQuantizeRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIQuantize").set_body_typed(MakeQnnCSIQuantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
