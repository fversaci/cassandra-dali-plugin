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
#include "./cassandra_dali.h"

namespace crs4 {

CassandraInteractive::CassandraInteractive(const dali::OpSpec &spec) :
  dali::InputOperator<dali::CPUBackend>(spec),
  batch_size(spec.GetArgument<int>("max_batch_size")),
  seed(spec.GetArgument<int64_t>("seed")),
  prefetch_buffers(spec.GetArgument<int>("prefetch_buffers")),
  slow_start(spec.GetArgument<int>("slow_start")),
  cloud_config(spec.GetArgument<std::string>("cloud_config")),
  cassandra_ips(spec.GetArgument<std::vector<std::string>>("cassandra_ips")),
  cassandra_port(spec.GetArgument<int>("cassandra_port")),
  table(spec.GetArgument<std::string>("table")),
  label_type(spec.GetArgument<std::string>("label_type")),
  label_col(spec.GetArgument<std::string>("label_col")),
  data_col(spec.GetArgument<std::string>("data_col")),
  id_col(spec.GetArgument<std::string>("id_col")),
  username(spec.GetArgument<std::string>("username")),
  password(spec.GetArgument<std::string>("password")),
  use_ssl(spec.GetArgument<bool>("use_ssl")),
  ssl_certificate(spec.GetArgument<std::string>("ssl_certificate")),
  ssl_own_certificate(spec.GetArgument<std::string>("ssl_own_certificate")),
  ssl_own_key(spec.GetArgument<std::string>("ssl_own_key")),
  ssl_own_key_pass(spec.GetArgument<std::string>("ssl_own_key_pass")),
  io_threads(spec.GetArgument<int>("io_threads")),
  copy_threads(spec.GetArgument<int>("copy_threads")),
  wait_threads(spec.GetArgument<int>("wait_threads")),
  comm_threads(spec.GetArgument<int>("comm_threads")),
  ooo(spec.GetArgument<bool>("ooo")),
  cow_dilute(slow_start -1) {
  DALI_ENFORCE(prefetch_buffers >= 0,
     "prefetch_buffers should be non-negative.");
  DALI_ENFORCE(label_type == "int" || label_type == "image" || label_type == "none",
     "label_type can only be int, image or none.");
  DALI_ENFORCE(slow_start >= 0,
     "slow_start should be either 0 (disabled) or >= 1 (prefetch dilution).");
  DALI_ENFORCE(batch_size * prefetch_buffers <= 32768 * io_threads,
     "please satisfy this constraint: batch_size * prefetch_buffers <= 32768 * io_threads");
  batch_ldr = new BatchLoader(table, label_type, label_col, data_col, id_col,
                        username, password, cassandra_ips, cassandra_port,
                        cloud_config, use_ssl, ssl_certificate,
                        ssl_own_certificate, ssl_own_key, ssl_own_key_pass,
                        io_threads, 1 + prefetch_buffers, copy_threads,
                        wait_threads, comm_threads, ooo);
}

void CassandraInteractive::prefetch_one() {
  // enforce max batch size
  DALI_ENFORCE(uuids.num_samples() <= batch_size,
       dali::make_string("batch_size must be <= ", batch_size, ", found ",
                      uuids.num_samples(), " samples."));
  // prepare and prefetch
  auto bs = uuids.num_samples();
  auto cass_uuids = std::vector<CassUuid>(bs);
  for (auto i=0; i != bs; ++i) {
    auto d_ptr = uuids[i].data<dali::uint64>();
    auto c_uuid = &cass_uuids[i];
    c_uuid->time_and_version = *(d_ptr++);
    c_uuid->clock_seq_and_node = *d_ptr;
  }
  batch_ldr->prefetch_batch(cass_uuids);
  ++curr_prefetch;
}

void CassandraInteractive::try_read_input(const dali::Workspace &ws) {
  if (HasDataInQueue()) {
    // forward input data to uuids tensorlist
    auto &thread_pool = ws.GetThreadPool();
    ForwardCurrentData(uuids, null_data_id, thread_pool);
    input_read = true;
  } else {
    input_read = false;
  }
}

bool CassandraInteractive::SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                          const dali::Workspace &ws) {
  uuids.Reset();
  uuids.set_pinned(false);
  try_read_input(ws);
  return false;
}

bool CassandraInteractive::ok_to_fill() {
  // fast start
  if (slow_start == 0)
    return true;
  // slow start: prefetch once every slow_start steps
  ++cow_dilute;
  cow_dilute %= slow_start;
  if (cow_dilute !=0) {
    return false;
  }
  return true;
}

void CassandraInteractive::fill_buffer(dali::Workspace &ws) {
  // start prefetching
  if (input_read) {
    prefetch_one();
    try_read_input(ws);
  }
}

void CassandraInteractive::fill_buffers(dali::Workspace &ws) {
  // start prefetching
  int num_buff = (slow_start > 0 && prefetch_buffers > 0) ? 1 : prefetch_buffers;
  for (int i=0; i < num_buff && ok_to_fill(); ++i) {
    fill_buffer(ws);
  }
}

void CassandraInteractive::RunImpl(dali::Workspace &ws) {
  // fill prefetch buffers
  if (curr_prefetch < prefetch_buffers) {
    fill_buffers(ws);
  }
  // if possible prefetch one before getting one
  if (input_read) {
    prefetch_one();
  }
  DALI_ENFORCE(curr_prefetch > 0, "No data ready to be retrieved. Have you prefetched?");
  BatchImgLab batch = batch_ldr->blocking_get_batch();
  // share features with output
  auto &features = ws.Output<dali::CPUBackend>(0);
  features.ShareData(batch.first);
  // share labels with output
  auto &labels = ws.Output<dali::CPUBackend>(1);
  labels.ShareData(batch.second);
  --curr_prefetch;
  SetDepletedOperatorTrace(ws, !(curr_prefetch > 0 || HasDataInQueue()));
}

CassandraSelfFeed::CassandraSelfFeed(const dali::OpSpec &spec) :
  CassandraInteractive(spec),
  source_uuids(spec.GetArgument<crs4::StrUUIDs>("source_uuids")),
  shard_id(spec.GetArgument<int>("shard_id")),
  num_shards(spec.GetArgument<int>("num_shards")),
  shuffle_after_epoch(spec.GetArgument<bool>("shuffle_after_epoch")),
  loop_forever(spec.GetArgument<bool>("loop_forever")) {
  DALI_ENFORCE(source_uuids.size() > 0,
               "plese provide a non-empty list of source_uuids");
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

CassandraUncoupled::CassandraUncoupled(const dali::OpSpec &spec) :
  CassandraInteractive(spec),
  mini_batch_size(spec.GetArgument<int>("mini_batch_size"))
{
  if (mini_batch_size<=0) {
    mini_batch_size = batch_size;
  }      
}

bool CassandraUncoupled::SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                           const dali::Workspace &ws) {
  // create mini batches from list
  if (curr_prefetch == 0 && HasDataInQueue()) {
    uuids.Reset();
    uuids.set_pinned(false);
    list_to_minibatches(ws);
  }
  return false;
}

void CassandraUncoupled::prefetch_one() {
  // exit if no data to prefetch
  if (input_interval >= intervals.size())
    return;
  // prepare and prefetch
  auto start = intervals[input_interval].first;
  auto end = intervals[input_interval].second;
  ++input_interval;
  auto bs = end-start;
  auto cass_uuids = std::vector<CassUuid>(bs);
  size_t ci=0;
  for (auto i=start; i != end; ++i, ++ci) {
    auto d_ptr = uuids[i].data<dali::uint64>();
    auto c_uuid = &cass_uuids[ci];
    c_uuid->time_and_version = *(d_ptr++);
    c_uuid->clock_seq_and_node = *d_ptr;
  }
  batch_ldr->prefetch_batch(cass_uuids);
  ++curr_prefetch;
}

void CassandraUncoupled::fill_buffers(dali::Workspace &ws) {
  // start prefetching
  int num_buff = (slow_start > 0 && prefetch_buffers > 0) ? 1 : prefetch_buffers;
  for (int i=0; i < num_buff && ok_to_fill(); ++i) {
    prefetch_one();
  }
}

void CassandraUncoupled::list_to_minibatches(const dali::Workspace &ws) {
  DALI_ENFORCE(HasDataInQueue(), "No UUIDs have been provided");
  // forward input data to uuids tensorlist
  auto &thread_pool = ws.GetThreadPool();
  ForwardCurrentData(uuids, null_data_id, thread_pool);  
  size_t full_sz = uuids.num_samples();
  intervals.clear();
  input_interval = 0;
  // split uuids in minibatches
  size_t floor_sz = mini_batch_size * (full_sz / mini_batch_size);
  for (size_t i = 0; i < floor_sz; i += mini_batch_size) {
    intervals.push_back(std::make_pair(i, i + mini_batch_size));
  }  
  // handle last batch
  if (floor_sz != full_sz) {
    intervals.push_back(std::make_pair(floor_sz, full_sz));
  }
}

void CassandraUncoupled::RunImpl(dali::Workspace &ws) {
  // fill prefetch buffers
  if (curr_prefetch < prefetch_buffers) {
    fill_buffers(ws);
  }
  // try to prefetch one minibatch
  prefetch_one();
  // consume data
  output = batch_ldr->blocking_get_batch();
  --curr_prefetch;
  // share features with output
  auto &features = ws.Output<dali::CPUBackend>(0);
  features.ShareData(output.first);
  // share labels with output
  auto &labels = ws.Output<dali::CPUBackend>(1);
  labels.ShareData(output.second);
  SetDepletedOperatorTrace(ws, !(curr_prefetch > 0 || HasDataInQueue()));
}

}  // namespace crs4

// register CassandraInteractive class

DALI_REGISTER_OPERATOR(crs4__cassandra_interactive, crs4::CassandraInteractive, dali::CPU);

DALI_SCHEMA(crs4__cassandra_interactive)
.DocStr("Reads UUIDs via feed_input and returns images and labels/masks")
.NumInput(0)
.NumOutput(2)
.AddOptionalArg<std::string>("cloud_config",
   R"(Cloud configuration for Cassandra (e.g., AstraDB))", "")
.AddOptionalArg("cassandra_ips",
   R"(List of Cassandra IPs)", std::vector<std::string>())
.AddOptionalArg("cassandra_port",
   R"(Port to connect to in the Cassandra server)", 9042)
.AddOptionalArg<std::string>("table", R"()", nullptr)
// label type: int (classification), image (segmentation mask), none
.AddOptionalArg<std::string>("label_type", R"()", "int")
.AddOptionalArg<std::string>("label_col", R"()", nullptr)
.AddOptionalArg<std::string>("data_col", R"()", nullptr)
.AddOptionalArg<std::string>("id_col", R"()", nullptr)
.AddOptionalArg<std::string>("username", R"()", nullptr)
.AddOptionalArg<std::string>("password", R"()", nullptr)
.AddOptionalArg("use_ssl", R"(Encrypt Cassandra connection with SSL)", false)
.AddOptionalArg<std::string>("ssl_certificate",
   R"(Optional SSL server certificate)", "")
.AddOptionalArg<std::string>("ssl_own_certificate",
   R"(Optional SSL client certificate)", "")
.AddOptionalArg<std::string>("ssl_own_key",
   R"(Optional SSL key)", "")
.AddOptionalArg<std::string>("ssl_own_key_pass",
   R"(Optional password for SSL key)", "")
.AddOptionalArg("prefetch_buffers", R"(Number of prefetch buffers)", 1)
.AddOptionalArg("io_threads",
   R"(Number of io threads used by the Cassandra driver)", 2)
.AddOptionalArg("copy_threads",
   R"(Number of threads copying data in parallel)", 2)
.AddOptionalArg("wait_threads", R"(Parallelism for waiting threads)", 2)
.AddOptionalArg("comm_threads", R"(Parallelism for communication threads)", 2)
.AddOptionalArg("blocking", R"(block until the data is available)", true)
.AddOptionalArg("no_copy", R"(should DALI copy the buffer when ``feed_input`` is called?)", false)
.AddOptionalArg("ooo", R"(Enable out-of-order batches)", false)
.AddOptionalArg("slow_start", R"(How much to dilute prefetching)", 0)
.AddParent("InputOperatorBase");

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
.AddOptionalArg("shuffle_after_epoch", R"(Reshuffling uuids at each epoch)",
   false)
.AddOptionalArg("loop_forever", R"(Loop on souce_uuids)", true)
.AddParent("crs4__cassandra_interactive");

// register CassandraUncoupled class

DALI_REGISTER_OPERATOR(crs4__cassandra_uncoupled, crs4::CassandraUncoupled, dali::CPU);

DALI_SCHEMA(crs4__cassandra_uncoupled)
.DocStr("Reads UUIDs as a large batch and returns images and labels/masks")
.NumInput(0)
.NumOutput(2)
.AddOptionalArg("mini_batch_size",
   R"code(Size of internal mini-batches.)code", -1)
.AddParent("crs4__cassandra_interactive");
