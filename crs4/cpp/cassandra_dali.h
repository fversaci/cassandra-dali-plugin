// Copyright 2021-2 CRS4
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef CRS4_CASSANDRA_DALI_H_
#define CRS4_CASSANDRA_DALI_H_

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <dali/pipeline/operator/operator.h>
#include <dali/operators/reader/reader_op.h>
#include "batch_loader.h"
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
    ret.epoch_size_padded = num_shards
      * std::ceil(uuids.size() / (double)num_shards);
    ret.number_of_shards = num_shards;
    ret.shard_id = shard_id;
    ret.pad_last_batch = true;
    ret.stick_to_shard = true;
    return ret;
  }

  ~Cassandra() override {
    if (bh!=nullptr) {
      delete bh;
    }
  }

protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<::dali::CPUBackend> &ws) override {
    return false;
  }

  inline void Reset() {
    current = shard_begin;
    current_epoch++;

    if (shuffle_after_epoch) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch);
      std::shuffle(uuids.begin(), uuids.end(), g);
    }
  }

  void RunImpl(::dali::workspace_t<dali::CPUBackend> &ws) override;

  void set_shard_sizes(){
    int dataset_size = uuids.size();
    shard_size = std::ceil(uuids.size() / (double)num_shards);
    int pos_begin = std::floor(shard_id*dataset_size/(double)num_shards);
    shard_begin = uuids.begin() + pos_begin;
    shard_end = shard_begin + shard_size;
  }

private:
  void prefetch_one();
  // variables
  std::vector<std::string> uuids;  
  std::vector<std::string> cassandra_ips;
  int cassandra_port;  
  std::string table;
  std::string label_col;
  std::string data_col;
  std::string id_col;
  std::string username;
  std::string password;
  BatchHandler* bh = nullptr;
  int batch_size;
  int prefetch_buffers;
  int io_threads;
  int copy_threads;
  int wait_threads;
  int comm_threads;
  bool use_ssl;
  std::string ssl_certificate;
  std::vector<std::string>::iterator current;
  bool shuffle_after_epoch;
  int current_epoch=-1;
  const int shard_id;
  const int num_shards;
  std::vector<std::string>::iterator shard_begin;
  std::vector<std::string>::iterator shard_end;
  int shard_size;
};

}  // namespace crs4

#endif  // CRS4_CASSANDRA_H_
