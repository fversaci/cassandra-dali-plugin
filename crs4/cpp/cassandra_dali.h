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

#ifndef CRS4_CPP_CASSANDRA_DALI_H_
#define CRS4_CPP_CASSANDRA_DALI_H_

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/operators/reader/reader_op.h"
#include "./batch_loader.h"

namespace crs4 {

using StrUUIDs = std::vector<std::string>;
using TL_UUIDs = dali::TensorList<dali::CPUBackend>;
  
class Cassandra : public dali::InputOperator<dali::CPUBackend> {
 public:
  explicit Cassandra(const dali::OpSpec &spec);

  Cassandra(const Cassandra&) = delete;
  Cassandra& operator=(const Cassandra&) = delete;
  Cassandra(Cassandra&&) = delete;
  Cassandra& operator=(Cassandra&&) = delete;

  ~Cassandra() override {
    if (batch_ldr != nullptr) {
      delete batch_ldr;
    }
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
    return 2;
  }


  dali::DALIDataType in_dtype() const override {
    return dali::DALIDataType::DALI_UINT64;
  }

 protected:
  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                 const dali::Workspace &ws) override;

  void RunImpl(dali::Workspace &ws) override;

 private:
  void prefetch_one(const dali::TensorList<dali::CPUBackend>&);
  void fill_buffer(dali::Workspace &ws);
  void fill_buffers(dali::Workspace &ws);
  bool ok_to_fill();
  // variables
  dali::TensorList<dali::CPUBackend> uuids;
  StrUUIDs source_uuids;
  TL_UUIDs tl_uuids;
  void convert_uuids();
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
  BatchLoader* batch_ldr = nullptr;
  int batch_size;
  int prefetch_buffers;
  int io_threads;
  int copy_threads;
  int wait_threads;
  int comm_threads;
  bool use_ssl;
  bool ooo;
  int slow_start;  // prefetch dilution
  int cow_dilute;  // counter for prefetch dilution
  int curr_prefetch = 0;
  std::string ssl_certificate;
  std::string ssl_own_certificate;
  std::string ssl_own_key;
  std::string ssl_own_key_pass;
  bool buffers_not_full = true;
  std::optional<std::string> null_data_id = std::nullopt;
  dali::TensorLayout in_layout_ = "B";  // Byte stream
};

}  // namespace crs4

#endif  // CRS4_CPP_CASSANDRA_DALI_H_
