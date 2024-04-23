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

#ifndef CRS4_CPP_CASSANDRA_DALI_SELFFEED_H_
#define CRS4_CPP_CASSANDRA_DALI_SELFFEED_H_

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cmath>
#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/operators/reader/reader_op.h"
#include "./cassandra_dali_interactive.h"
#include "./batch_loader.h"

namespace crs4 {

using StrUUIDs = std::vector<std::string>;
using U64_UUIDs = std::vector<std::pair<int64_t, int64_t>>;

class CassandraSelfFeed : public CassandraInteractive {
 public:
  explicit CassandraSelfFeed(const dali::OpSpec &spec);

  dali::ReaderMeta GetReaderMeta() const override {
    dali::ReaderMeta ret;
    ret.epoch_size = source_uuids.size();
    ret.epoch_size_padded = num_shards
      * std::ceil(ret.epoch_size / static_cast<double>(num_shards));
    ret.number_of_shards = num_shards;
    ret.shard_id = shard_id;
    ret.pad_last_batch = true;
    ret.stick_to_shard = true;
    return ret;
  }

 protected:
  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                 const dali::Workspace &ws) override;
  void feed_new_epoch() {
    current_epoch++;
    if (shuffle_every_epoch) {
      std::mt19937 g(seed + current_epoch);
      std::shuffle(u64_uuids.begin(), u64_uuids.end(), g);
    }
    feed_epoch();
    left_batches = batches_per_epoch;
  }

 private:
  void set_shard_sizes() {
    size_t dataset_size = u64_uuids.size();
    shard_size = std::ceil(dataset_size / static_cast<double>(num_shards));
    batches_per_epoch = std::ceil(shard_size
                                  / static_cast<double>(batch_size));
    pad_shard_size = batches_per_epoch * batch_size;
    size_t pos_begin = std::floor(shard_id * dataset_size
                               / static_cast<double>(num_shards));
    shard_begin = u64_uuids.begin() + pos_begin;
    shard_end = shard_begin + shard_size;
  }

  StrUUIDs source_uuids;
  U64_UUIDs u64_uuids;
  int current_epoch = -1;
  const int shard_id;
  const int num_shards;
  bool shuffle_every_epoch;
  bool loop_forever;
  U64_UUIDs::iterator shard_begin;
  U64_UUIDs::iterator shard_end;
  size_t shard_size;
  size_t batches_per_epoch;
  size_t pad_shard_size;
  size_t left_batches = 0;
  void convert_uuids();
  void feed_epoch();
};

}  // namespace crs4

#endif  // CRS4_CPP_CASSANDRA_DALI_SELFFEED_H_
