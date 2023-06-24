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
 * \file src/relay/backend/contrib/csinn/xuantie_aot.h
 * \brief The base class for xuantie_aot.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_XUANTIE_AOT_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_XUANTIE_AOT_H_

#include <map>
#include <string>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

class GraphPartition : public HHBExprVisitor {
 public:
  GraphPartition() {}
  virtual void graph_partition(const Expr& expr);
  Array<Expr> get_partitions(void) { return partitions_; }

 protected:
  void create_partition(const CallNode* call);
  virtual void visit_expr(const CallNode* call);

 private:
  Expr expr_;
  Array<Expr> partitions_;
};

class XuanTie_AOT {
 public:
  explicit XuanTie_AOT(string quantization_scheme) {
    if (quantization_scheme == "float32") {
      base_dtype = DataType::Float(32);
    } else if (quantization_scheme == "float16") {
      base_dtype = DataType::Float(16);
    } else {
      LOG(FATAL) << "AoT Unsupported scheme: " << quantization_scheme;
    }
  }
  virtual void generate_c_intrinsic(void);
  virtual void compile(const Expr& expr);
  /* return intrinsic */
  virtual string get_ccode(void);
  /* TODO: return binary */
  // virtual void *get_binary(void);
 private:
  RelayExpr tvm_realy_expr(const RelayExpr& expr, string* name);

  Expr expr_;
  ToRelay export_relay_;
  Array<Expr> partitions_;
  GraphPartition* graph_partition_;
  std::map<string, string> c_intrinsic_;

  DataType base_dtype;
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_XUANTIE_AOT_H_
