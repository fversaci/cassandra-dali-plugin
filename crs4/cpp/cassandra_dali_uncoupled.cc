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
#include "./cassandra_dali_uncoupled.h"

namespace crs4 {

CassandraUncoupled::CassandraUncoupled(const dali::OpSpec &spec) :
  CassandraInteractive(spec),
  mini_batch_size(spec.GetArgument<int>("mini_batch_size")) {
  if (mini_batch_size <= 0) {
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
  size_t ci = 0;
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

// register CassandraUncoupled class

DALI_REGISTER_OPERATOR(crs4__cassandra_uncoupled, crs4::CassandraUncoupled, dali::CPU);

DALI_SCHEMA(crs4__cassandra_uncoupled)
.DocStr("Reads UUIDs as a large batch and returns images and labels/masks")
.NumInput(0)
.NumOutput(2)
.AddOptionalArg("mini_batch_size",
   R"code(Size of internal mini-batches.)code", -1)
.AddParent("crs4__cassandra_interactive");
