// Copyright 2021-2 CRS4
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef CRS4_CPP_CASSANDRA_DALI_H_
#define CRS4_CPP_CASSANDRA_DALI_H_

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/reader/reader_op.h"
#include "./batch_loader.h"

namespace crs4 {

class Cassandra : public ::dali::Operator<::dali::CPUBackend> {
 public:
  explicit Cassandra(const ::dali::OpSpec &spec);

  Cassandra(const Cassandra&) = delete;
  Cassandra& operator=(const Cassandra&) = delete;
  Cassandra(Cassandra&&) = delete;
  Cassandra& operator=(Cassandra&&) = delete;

  ~Cassandra() override {
    if (batch_ldr != nullptr) {
      delete batch_ldr;
    }
  }

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::Workspace &ws) override {
    return false;
  }

  void RunImpl(::dali::Workspace &ws) override;

 private:
  void prefetch_one(const dali::TensorList<dali::CPUBackend>&);
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
  BatchLoader* batch_ldr = nullptr;
  int batch_size;
  int prefetch_buffers;
  int io_threads;
  int copy_threads;
  int wait_threads;
  int comm_threads;
  bool use_ssl;
  std::string ssl_certificate;
  std::vector<std::string>::iterator current;
};

}  // namespace crs4

#endif  // CRS4_CPP_CASSANDRA_DALI_H_
