// Copyright 2021-2 CRS4
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef CRS4_CASSANDRA_H_
#define CRS4_CASSANDRA_H_

#include <vector>
#include <map>
#include <string>
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/reader/reader_op.h"
#include "batchhandler.h"
namespace crs4 {

class Cassandra : public ::dali::Operator<::dali::CPUBackend> {
public:
  explicit Cassandra(const ::dali::OpSpec &spec);

  Cassandra(const Cassandra&) = delete;
  Cassandra& operator=(const Cassandra&) = delete;
  Cassandra(Cassandra&&) = delete;
  Cassandra& operator=(Cassandra&&) = delete;

  ::dali::ReaderMeta GetReaderMeta() const override {
    ::dali::ReaderMeta ret;
    ret.epoch_size = uuids.size();
    ret.epoch_size_padded = uuids.size();
    ret.number_of_shards = 1;
    ret.shard_id = 0;
    ret.pad_last_batch = true;
    ret.stick_to_shard = true;
    return ret;
  }

  ~Cassandra() override {
    if (bh!=nullptr){
      delete bh;
    }
  }

protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<::dali::CPUBackend> &ws) override {
    return false;
  }

  inline void Reset() {
    // std::cout << "Reset" << std::endl;
    current = uuids.begin();
  }
  
  void RunImpl(::dali::workspace_t<dali::CPUBackend> &ws) override;

private:
  void prefetch_one();

  std::vector<std::string> uuids;
  std::vector<std::string> cass_ips;
  int cass_port;
  std::vector<std::string> cass_conf;
  BatchHandler* bh = nullptr;
  int batch_size;
  int prefetch_buffers;
  int tcp_connections;
  int copy_threads;
  int wait_par;
  int comm_par;
  bool use_ssl;
  std::vector<std::string>::iterator current;
};

}  // namespace crs4

#endif  // CRS4_CASSANDRA_H_
