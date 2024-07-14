// Copyright 2022 CRS4 (http://www.crs4.it/)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRS4_CPP_CASSANDRA_DALI_INTERACTIVE_H_
#define CRS4_CPP_CASSANDRA_DALI_INTERACTIVE_H_

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cmath>
#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/operators/reader/reader_op.h"
#include "./batch_loader.h"

namespace crs4 {

class CassandraInteractive : public dali::InputOperator<dali::CPUBackend> {
 public:
  explicit CassandraInteractive(const dali::OpSpec &spec);

  CassandraInteractive(const CassandraInteractive&) = delete;
  CassandraInteractive& operator=(const CassandraInteractive&) = delete;
  CassandraInteractive(CassandraInteractive&&) = delete;
  CassandraInteractive& operator=(CassandraInteractive&&) = delete;

  ~CassandraInteractive() override {
    if (batch_ldr != nullptr) {
      delete batch_ldr;
    }
  }

  /*
  bool Setup(std::vector<dali::OutputDesc> &output_desc,
             const dali::Workspace &ws) override {
    EnforceUniformInputBatchSize<dali::CPUBackend>(ws);
    CheckInputLayouts(ws, spec_);
    return SetupImpl(output_desc, ws);
  }
  */

  void Run(dali::Workspace &ws) override {
    // SetupSharedSampleParams(ws);
    RunImpl(ws);
    ws.GetThreadPool().WaitForWork();
    // EnforceUniformOutputBatchSize<dali::CPUBackend>(ws);
  }

  int NextBatchSize() override {
    return batch_size;
  }

  void Advance() override {
  }

  const dali::TensorLayout& in_layout() const override {
    return in_layout_;
  }

  int in_ndim() const override {
    return 1;
  }

  dali::DALIDataType in_dtype() const override {
    return dali::DALIDataType::DALI_UINT64;
  }

 protected:
  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                 const dali::Workspace &ws) override;
  void RunImpl(dali::Workspace &ws) override;
  std::optional<std::string> null_data_id = std::nullopt;
  int batch_size;
  int64_t seed;
  BatchLoader* batch_ldr = nullptr;
  dali::TensorList<dali::CPUBackend> uuids;
  size_t curr_prefetch = 0;
  size_t prefetch_buffers;
  int slow_start;  // prefetch dilution
  bool ok_to_fill();

 private:
  void prefetch_one();
  void fill_buffer(dali::Workspace &ws);
  void fill_buffers(dali::Workspace &ws);
  void try_read_input(const dali::Workspace &ws);
  // variables
  std::string cloud_config;
  std::vector<std::string> cassandra_ips;
  int cassandra_port;
  std::string table;
  std::string label_type;
  std::string label_col;
  std::string data_col;
  std::string id_col;
  std::string username;
  std::string password;
  bool use_ssl;
  std::string ssl_certificate;
  std::string ssl_own_certificate;
  std::string ssl_own_key;
  std::string ssl_own_key_pass;
  size_t io_threads;
  size_t copy_threads;
  size_t wait_threads;
  size_t comm_threads;
  bool ooo;
  int cow_dilute;  // counter for prefetch dilution
  bool input_read = false;
  dali::TensorLayout in_layout_ = "B";  // Byte stream
};

}  // namespace crs4

#endif  // CRS4_CPP_CASSANDRA_DALI_INTERACTIVE_H__
