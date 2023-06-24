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
 * \file src/relay/backend/contrib/csinn/ref.cc
 * \brief Implementation of CSINN codegen APIs.
 */

#include "ref.h"

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

void CodegenRef::EmitReadParams() {
  func_def_.OneLine("static char* read_params(char* path) {");
  func_def_.EnterScope();
  func_def_.OneLine("FILE* fp = fopen(path, \"rb\");");
  func_def_.OneLine("if (fp == NULL) {printf(\"cannot open params\\n\");}");
  func_def_.OneLine("fseek(fp, 0, SEEK_END);");
  func_def_.OneLine("int file_size = ftell(fp);");
  func_def_.OneLine("rewind(fp);");
  func_def_.OneLine("char* buffer = (char*)malloc(file_size);");
  func_def_.OneLine("if (buffer == NULL) {return NULL;}");
  func_def_.OneLine("int ret = fread(buffer, 1, file_size, fp);");
  func_def_.OneLine("if (ret != file_size) {return NULL;}");
  func_def_.OneLine("fclose(fp);");
  func_def_.OneLine("return buffer;");
  func_def_.ExitScope();
  func_def_.OneLine("}");
}

void CodegenRef::GenerateBackendCFunc(const string& func_name, const Array<Var>& args,
                                      const output_element& out) {
  func_def_.NewLine();
  std::ostringstream t0;
  string in_dtype = cfg->dtype_input;
  string weight_dtype = cfg->dtype_weight;
  func_def_.NewLine();
  EmitReadParams();
  t0 << "int " << func_name << "_runtime_wrapper_(";
  for (uint i = 0; i < args.size(); i++) {
    t0 << "void *inputs" << i << ", ";
  }
  for (uint i = 0; i < output_list_.size() - 1; i++) {
    t0 << "void *outputs" << i << ", ";
  }

  t0 << "void *outputs" << output_list_.size() - 1 << ") {";
  func_def_.OneLine(t0);

  func_def_.EnterScope();

  func_def_.OneLine("static int has_opened;");
  func_def_.OneLine("static char *params_base = NULL;");
  func_def_.OneLine("if (has_opened == 0) {");
  t0 << "params_base = read_params(\"" << params_path_ << "\");";
  func_def_.OneLine(t0);
  func_def_.OneLine("has_opened = 1;");
  func_def_.OneLine("}");

  string out_dtype = GetCSINNDtype(weight_dtype);

  t0 << "struct csinn_tensor* qinput_tensors[" << args.size() << "];";
  func_def_.OneLine(t0);

  for (uint i = 0; i < args.size(); i++) {
    const auto& dtype_str = GetDtypeString(args[i]);
    std::string new_name = replace(args[i]->name_hint());
    auto input_tensor = bm_graph.get_tensor(new_name);
    CHECK(input_tensor);
    auto ishape = args[i]->get_shape();
    int size = 1;
    if (weight_dtype == "float16" || weight_dtype == "bfloat16" || weight_dtype == "int16_t") {
      size = size * 2;
      weight_dtype = "int16_t";
    } else if (weight_dtype == "float" || weight_dtype == "float32") {
      size = size * 4;
    }
    for (uint j = 0; j < ishape.size(); j++) {
      size = size * ishape[j];
    }
    if (dtype_str == "float" || dtype_str == "int32_t") {
      t0 << weight_dtype << "* __" << new_name << " = (" << weight_dtype << "*)malloc(" << size
         << ");";
      func_def_.OneLine(t0);
      string in_name = "(" + dtype_str + "*)inputs" + to_string(i) + ";";

      string in_tensor = "input_" + to_string(i);
      t0 << "struct csinn_tensor *" << in_tensor << " = csinn_alloc_tensor(NULL);";
      func_def_.OneLine(t0);
      t0 << in_tensor << "->data = " << in_name << ";";
      func_def_.OneLine(t0);
      t0 << in_tensor << "->dim_count = " << ishape.size() << ";";
      func_def_.OneLine(t0);
      for (uint j = 0; j < ishape.size(); j++) {
        t0 << in_tensor << "->dim[" << j << "] = " << ishape[j] << ";";
        func_def_.OneLine(t0);
      }

      string qin_tensor = "qinput_" + to_string(i);
      func_def_.OneLine(t0);
      t0 << "struct csinn_tensor *" << qin_tensor << " = csinn_alloc_tensor(NULL);";
      func_def_.OneLine(t0);
      t0 << "csinn_tensor_copy(" << qin_tensor << ", " << in_tensor << ");";
      func_def_.OneLine(t0);
      t0 << qin_tensor << "->data = __" << new_name << ";";
      func_def_.OneLine(t0);

      t0 << qin_tensor << "->qinfo = (struct csinn_quant_info *)(params_base + "
         << input_tensor->get_binary_model_qinfo_real_start() << ");";
      func_def_.OneLine(t0);
      t0 << qin_tensor << "->quant_channel =  " << input_tensor->tensor->quant_channel << ";";
      func_def_.OneLine(t0);
      t0 << qin_tensor << "->dtype = " << out_dtype << ";";
      func_def_.OneLine(t0);
      t0 << "shl_ref_nn_init(" << in_tensor << ", " << qin_tensor << ");";
      func_def_.NewLine();
      func_def_.OneLine(t0);
      t0 << "qinput_tensors[" << i << "] = " << qin_tensor << ";";
      func_def_.OneLine(t0);
    } else if (dtype_str == "uint8_t" || dtype_str == "int8_t" || dtype_str == "int64_t" ||
               dtype_str == "bool") {
      t0 << weight_dtype << "* __" << new_name
         << " = (" + weight_dtype + "*)inputs" + to_string(i) + ";";
      func_def_.OneLine(t0);
      string qin_tensor = "qinput_" + to_string(i);
      t0 << "struct csinn_tensor *" << qin_tensor << " = csinn_alloc_tensor(NULL);";
      func_def_.OneLine(t0);
      t0 << qin_tensor << "->data = __" << new_name << ";";
      func_def_.OneLine(t0);
      t0 << qin_tensor << "->dim_count = " << ishape.size() << ";";
      func_def_.OneLine(t0);
      for (uint j = 0; j < ishape.size(); j++) {
        t0 << qin_tensor << "->dim[" << j << "] = " << ishape[j] << ";";
        func_def_.OneLine(t0);
      }
      t0 << "qinput_tensors[" << i << "] = " << qin_tensor << ";";
      func_def_.OneLine(t0);
    } else {
      LOG(ERROR) << "get error dtype: " << dtype_str;
    }
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    uint out_dim_count = output_list_[i].shape.size();
    int size = output_list_[i].size;
    if (output_list_[i].dtype == "float" || output_list_[i].dtype == "int32_t") {
      size *= 4;
    } else if (output_list_[i].dtype == "float16" || output_list_[i].dtype == "bfloat16" ||
               output_list_[i].dtype == "int16_t") {
      size *= 2;
      output_list_[i].dtype = "int16_t";
    }
    if (output_list_[i].call != NULL) {
      auto output_tensor = bm_graph.get_tensor(output_list_[i].name);
      CHECK(output_tensor);

      t0 << output_list_[i].dtype << "* out_q_" << i << " = (" << output_list_[i].dtype
         << " *)malloc(" << size << ");";
      func_def_.NewLine();
      func_def_.OneLine(t0);
      /* FIXME: real output type */
      auto out_dtype = "float";
      // backend::DType2String(backend::GetType(output_list_[i].call->checked_type()));
      string out_tensor = "output_" + to_string(i);
      t0 << "struct csinn_tensor *" << out_tensor << " = csinn_alloc_tensor(NULL);";
      func_def_.OneLine(t0);
      t0 << out_tensor << "->data = "
         << "outputs" << i << ";";
      func_def_.OneLine(t0);
      t0 << out_tensor << "->dtype = " << GetCSINNDtype(out_dtype) << ";";
      func_def_.OneLine(t0);
      t0 << out_tensor << "->layout = " << output_tensor->to_layout() << ";";
      func_def_.NewLine();
      func_def_.OneLine(t0);

      string qout_tensor = "qoutput_" + to_string(i);
      t0 << "struct csinn_tensor *" << qout_tensor << " = csinn_alloc_tensor(NULL);";
      func_def_.OneLine(t0);
      t0 << qout_tensor << "->data = out_q_" << i << ";";
      func_def_.OneLine(t0);
      uint dim_count = out_dim_count == 0 ? 1 : out_dim_count;
      t0 << qout_tensor << "->dim_count = " << dim_count << ";";
      func_def_.OneLine(t0);
      if (out_dim_count == 0) {
        t0 << qout_tensor << "->dim[" << 0 << "] = 1;";
        func_def_.OneLine(t0);
      }
      for (uint j = 0; j < out_dim_count; j++) {
        t0 << qout_tensor << "->dim[" << j << "] = " << output_tensor->tensor->dim[j] << ";";
        func_def_.OneLine(t0);
      }
      t0 << qout_tensor << "->qinfo = (struct csinn_quant_info *)(params_base + "
         << output_tensor->get_binary_model_qinfo_real_start() << ");";
      func_def_.OneLine(t0);
      t0 << qout_tensor << "->quant_channel = " << output_tensor->tensor->quant_channel << ";";
      func_def_.OneLine(t0);
      t0 << qout_tensor << "->dtype = " << GetCSINNDtype(output_list_[i].dtype) << ";";
      func_def_.OneLine(t0);
      t0 << qout_tensor << "->layout = " << output_tensor->to_layout() << ";";
      func_def_.OneLine(t0);
    } else {
      t0 << in_dtype << " *out_" << i << " = (" << in_dtype << " *)outputs" << i << ";";
      func_def_.OneLine(t0);
    }

    func_def_.NewLine();
    func_def_.OneLine(t0);
  }

  t0 << "struct csinn_session *sess = csinn_(params_base);\n";
  func_def_.OneLine(t0);

  t0 << "csinn_update_input_and_run(qinput_tensors, sess);\n";
  func_def_.OneLine(t0);

  for (uint i = 0; i < output_list_.size(); i++) {
    auto out_node = output_list_[i].call;
    if (out_node != NULL) {
      t0 << "  csinn_get_output(" << i << ", qoutput_" << i << ", sess);";
      func_def_.OneLine(t0);
      t0 << "struct csinn_tensor *foutput_" << i << " = shl_ref_tensor_transform_f32(qoutput_" << i
         << ");";
      func_def_.OneLine(t0);
      t0 << "memcpy(outputs" << i << ", foutput_" << i << "->data, csinn_tensor_byte_size(foutput_"
         << i << "));\n";
      func_def_.OneLine(t0);
      t0 << "shl_ref_tensor_transform_free_f32(foutput_" << i << ");\n";
      func_def_.OneLine(t0);
    }
  }

  func_def_.OneLine(t0);
  func_def_.OneLine("return 0;");
  func_def_.ExitScope();
  func_def_.OneLine("}");
  func_def_.OneLine("#include <tvm/runtime/c_runtime_api.h>");

  t0 << "int " << func_name << "_wrapper_(";
  func_def_.OneLine(t0);
  for (size_t i = 0; i < args.size(); i++) {
    t0 << "DLTensor* arg" << i << ",";
    func_def_.OneLine(t0);
  }
  for (size_t i = 0; i < output_list_.size() - 1; i++) {
    t0 << "DLTensor* out" << i << ",";
    func_def_.OneLine(t0);
  }
  t0 << "DLTensor* out" << output_list_.size() - 1 << ") {";
  func_def_.OneLine(t0);
  t0 << func_name << "_runtime_wrapper_(";
  for (size_t i = 0; i < args.size(); i++) {
    const auto& dtype_str = GetDtypeString(args[i]);
    t0 << "(" << dtype_str << "*)(arg" << i << "->data),\n";
  }
  for (size_t i = 0; i < output_list_.size() - 1; i++) {
    t0 << "(" << output_list_[i].dtype << "*)(out" << i << "->data),\n";
  }
  t0 << "(" << output_list_.back().dtype << "*)(out" << output_list_.size() - 1 << "->data));";
  func_def_.OneLine(t0);
  func_def_.OneLine("return 0;");
  func_def_.OneLine("}");

  func_def_.OneLine("#ifdef __cplusplus");
  func_def_.OneLine("extern \"C\" {");
  func_def_.OneLine("#endif");
  t0 << "TVM_DLL int32_t ";
  t0 << ext_func_id_ << "(";
  t0 << "TVMValue* args, ";
  t0 << "int* type_code, ";
  t0 << "int num_args, ";
  t0 << "TVMValue* out_value, ";
  t0 << "int* out_type_code) {";
  func_def_.OneLine(t0);
  func_def_.EnterScope();
  for (size_t i = 0; i < args.size(); i++) {
    t0 << "DLTensor* arg" << i << " = ";
    t0 << "(DLTensor*)(((TVMValue*)args)[" << i << "].v_handle);";
    func_def_.OneLine(t0);
  }
  for (size_t i = 0; i < output_list_.size(); i++) {
    t0 << "DLTensor* ret" << args.size() + i << " = ";
    t0 << "(DLTensor*)(((TVMValue*)args)[" << args.size() + i << "].v_handle);";
    func_def_.OneLine(t0);
  }

  t0 << func_name << "_wrapper_(";
  for (size_t i = 0; i < args.size(); i++) {
    t0 << "arg" << i << ",";
  }
  for (size_t i = 0; i < output_list_.size() - 1; i++) {
    t0 << "ret" << args.size() + i << ",";
  }
  t0 << "ret" << args.size() + output_list_.size() - 1 << ");";
  func_def_.OneLine(t0);
  func_def_.OneLine("return 0;");
  func_def_.ExitScope();
  func_def_.OneLine("}");
  func_def_.OneLine("#ifdef __cplusplus");
  func_def_.OneLine("}");
  func_def_.OneLine("#endif");
}

void CodegenRef::EmitHeader(void) {
  std::ostringstream t0;
  func_def_.OneLine("#include <shl_ref.h>");
  func_def_.NewLine();
}

string CodegenRef::get_ccode(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  EmitSessionRun();
  GenerateBackendCFunc(ext_func_id_, ext_func_args_, out_[0]);
  DumpConstant();
  return func_def_.str();
}

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
