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
#include "./cassandra_dali_interactive.h"

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
  DALI_ENFORCE(label_type == "int" || label_type == "blob" || label_type == "none",
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
    auto d_ptr = uuids[i].data<uint64_t>();
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

