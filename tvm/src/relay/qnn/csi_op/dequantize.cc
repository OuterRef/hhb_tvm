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
 * \file src/relay/qnn/op/dequantize.cc
 * \brief QNN dequantize operator. Dequantize operator converts from quantized
 * domain to unquantized domain.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(QnnCSIDequantizeAttrs);

bool QnnCSIDequantizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) {
    return false;
  }

  const auto input_dtype = data->dtype;
  ICHECK(input_dtype == DataType::Int(8) || input_dtype == DataType::UInt(8) ||
         input_dtype == DataType::Int(16) || input_dtype == DataType::Int(32))
      << "Input type should be one of the quantized types [unit8, int8, int16, int32] but was "
      << input_dtype;

  const auto* dequantize_attrs = attrs.as<QnnCSIDequantizeAttrs>();
  int axis = dequantize_attrs->axis;
  auto rank = static_cast<int>(data->shape.size());
  axis = (axis < 0) ? ((rank > 0) ? data->shape.size() + axis : 0) : axis;

  // If zero point and scale are scalar or have arbitrary rank with one element,
  // then axis doesn't matter.
  bool scale_is_scalar = (types[1].as<TensorTypeNode>())->shape.size() == 0 ||
                         get_const_int((types[1].as<TensorTypeNode>())->Size()) == 1;
  bool zp_is_scalar = (types[2].as<TensorTypeNode>())->shape.size() == 0 ||
                      get_const_int((types[2].as<TensorTypeNode>())->Size()) == 1;

  if (!scale_is_scalar || !zp_is_scalar) {
    ICHECK_LT(axis, rank > 0 ? rank : 1) << "axis " << dequantize_attrs->axis << " is out of range";
    ICHECK_GE(axis, 0) << "axis " << dequantize_attrs->axis << " is out of range";
  }

  PrimExpr axis_shape;
  if (!scale_is_scalar || !zp_is_scalar) {
    axis_shape = data->shape[axis];
  } else {
    axis_shape = Integer(1);
  }
  // Check and assign types for scale and zero points.
  AssignType(types[1], DataType::Float(32), axis_shape, reporter);  // scale
  AssignType(types[2], DataType::Int(32), axis_shape, reporter);    // zero point
  const Array<tvm::PrimExpr> oshape = data->shape;
  // assign output type, output will always be float 32.
  reporter->Assign(types[3], TensorType(oshape, DataType::Float(32)));
  return true;
}

Expr MakeQnnCSIDequantize(Expr data, Expr input_scale, Expr input_zero_point, int axis,
                          DataType out_dtype, Array<Array<IndexExpr>> q_params, String layer_name) {
  // real_value = scale * (quantized_value - zero_point)
  // A more detailed explanation can be found here -
  // https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  auto attrs = make_object<QnnCSIDequantizeAttrs>();

  attrs->axis = axis;
  attrs->out_dtype = std::move(out_dtype);
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);
  static const Op& op = Op::Get("qnn.csi.dequantize");
  return Call(op, {data, input_scale, input_zero_point}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.dequantize")
    .describe(R"code(Dequantizes the input and produces float32 output.
The input is always quantized (int8, uint8) and will be converted to float32 given input scale and zero_point.
- **data**: Quantized tensor of any shape to dequantize. The input data can be of floating point
)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIDequantizeAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The tensor to dequantize.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIDequantize", QnnCSIDequantizeRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIDequantize").set_body_typed(MakeQnnCSIDequantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
