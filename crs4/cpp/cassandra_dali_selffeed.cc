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

#include <iostream>
#include <fstream>
#include "./cassandra_dali_selffeed.h"

namespace crs4 {

CassandraSelfFeed::CassandraSelfFeed(const dali::OpSpec &spec) :
  CassandraInteractive(spec),
  source_uuids(spec.GetArgument<crs4::StrUUIDs>("source_uuids")),
  shard_id(spec.GetArgument<int>("shard_id")),
  num_shards(spec.GetArgument<int>("num_shards")),
  shuffle_every_epoch(spec.GetArgument<bool>("shuffle_every_epoch")),
  loop_forever(spec.GetArgument<bool>("loop_forever")) {
  DALI_ENFORCE(source_uuids.size() > 0,
               "please provide a non-empty list of source_uuids");
  DALI_ENFORCE(num_shards > shard_id,
               "num_shards needs to be greater than shard_id");
  convert_uuids();
  set_shard_sizes();
  // feed first epoch
  feed_new_epoch();
  if (loop_forever) {
    // feed also second epoch
    feed_new_epoch();
  }
}

bool CassandraSelfFeed::SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                           const dali::Workspace &ws) {
  // refeed uuids at the end of the epoch
  if (--left_batches == 0 && loop_forever) {
    feed_new_epoch();
  }
  return CassandraInteractive::SetupImpl(output_desc, ws);
}

void CassandraSelfFeed::feed_epoch() {
  // set up tensorlist buffer for batches
  std::vector<int64_t> v_sz(batch_size, 2);
  dali::TensorListShape t_sz(v_sz, batch_size, 1);
  auto tl_batch = dali::TensorList<dali::CPUBackend>();
  tl_batch.set_pinned(false);
  tl_batch.Resize(t_sz, dali::DALIDataType::DALI_UINT64);
  // feed batches
  auto it = shard_begin;
  size_t i = 0;
  int num;
  while (i < pad_shard_size - batch_size) {
    for (num = 0; num != batch_size ; ++i, ++num, ++it) {
      auto ten = (dali::uint64*) tl_batch.raw_mutable_tensor(num);
      ten[0] = it->first;
      ten[1] = it->second;
    }
    SetDataSource(tl_batch);  // feed batch
  }
  // handle last batch
  // 1) fill what remains
  auto last_elem = shard_begin;
  for (num = 0; i < shard_size; ++i, ++num, ++it) {
    auto ten = (dali::uint64*) tl_batch.raw_mutable_tensor(num);
    ten[0] = it->first;
    ten[1] = it->second;
    last_elem = it;
  }
  // 2) pad using last element
  for (; i < pad_shard_size; ++i, ++num) {
    auto ten = (dali::uint64*) tl_batch.raw_mutable_tensor(num);
    ten[0] = last_elem->first;
    ten[1] = last_elem->second;
  }
  SetDataSource(tl_batch);  // feed last batch
}

void CassandraSelfFeed::convert_uuids() {
  auto sz = source_uuids.size();
  u64_uuids.resize(sz);
  int num = 0;
  for (auto id = source_uuids.begin(); id != source_uuids.end(); ++id, ++num) {
    CassUuid cuid;
    cass_uuid_from_string(id->c_str(), &cuid);
    u64_uuids[num] = std::make_pair(cuid.time_and_version, cuid.clock_seq_and_node);
  }
}

}  // namespace crs4

// register CassandraSelfFeed class

DALI_REGISTER_OPERATOR(crs4__cassandra, crs4::CassandraSelfFeed, dali::CPU);

DALI_SCHEMA(crs4__cassandra)
.DocStr("Reads UUIDs via source_uuids and returns images and labels/masks")
.NumInput(0)
.NumOutput(2)
.AddOptionalArg("source_uuids", R"(Full list of uuids)",
   std::vector<std::string>())
.AddOptionalArg("num_shards",
   R"code(Partitions the data into the specified number of shards.
This is typically used for distributed training.)code", 1)
.AddOptionalArg("shard_id",
   R"code(Index of the shard to read.)code", 0)
.AddOptionalArg("shuffle_every_epoch", R"(Reshuffling uuids at each epoch)",
   false)
.AddOptionalArg("loop_forever", R"(Loop on souce_uuids)", true)
.AddParent("crs4__cassandra_interactive");

