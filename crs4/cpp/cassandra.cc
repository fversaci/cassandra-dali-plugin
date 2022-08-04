// Copyright 2021-2 CRS4
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <iostream>
#include <fstream>

#include "cassandra.h"

namespace crs4 {

Cassandra::Cassandra(const ::dali::OpSpec &spec) :
  ::dali::Operator<dali::CPUBackend>(spec),
  batch_size(spec.GetArgument<int>("max_batch_size")),
  uuids(spec.GetArgument<std::vector<std::string>>("uuids")),
  cass_ips(spec.GetArgument<std::vector<std::string>>("cass_ips")),
  cass_conf(spec.GetArgument<std::vector<std::string>>("cass_conf")),
  prefetch_buffers(spec.GetArgument<int>("prefetch_queue_depth")),
  tcp_connections(spec.GetArgument<int>("tcp_connections")),
  copy_threads(spec.GetArgument<int>("copy_threads"))
{
  bh = new BatchHandler(cass_conf[0], cass_conf[1], cass_conf[2],
                        cass_conf[3], cass_conf[4], cass_conf[5],
                        cass_ips, prefetch_buffers, tcp_connections,
                        copy_threads);
  current = uuids.begin();
  // start prefetching
  for (int i=0; i<prefetch_buffers; ++i)
    prefetch_one();
}

void Cassandra::prefetch_one() {
  auto dist = std::distance(current, uuids.end());
  // if reached the end, rewind
  if (dist==0){
    current = uuids.begin();
    dist = uuids.size();
  }
  // full batch
  if (dist>=batch_size) {
    auto batch_ids = std::vector(current, current+batch_size);
    current += batch_size;
    bh->prefetch_batch(batch_ids);
    return;
  }
  // pad partial batch
  auto batch_ids = std::vector(current, uuids.end());
  for (int i=dist; i!=batch_size; ++i)
    batch_ids.push_back(*uuids.rbegin());
  current = uuids.end();
  bh->prefetch_batch(batch_ids);
}

void Cassandra::RunImpl(::dali::HostWorkspace &ws) {
  BatchImgLab batch = bh->blocking_get_batch();
  prefetch_one();
  // copy features to output
  auto &features = ws.Output<::dali::CPUBackend>(0);
  features = std::move(batch.first);
  // copy labels to output
  auto &labels = ws.Output<::dali::CPUBackend>(1);
  labels = std::move(batch.second);
}

}  // namespace crs4

DALI_REGISTER_OPERATOR(crs4__cassandra,
                       ::crs4::Cassandra, ::dali::CPU);

DALI_SCHEMA(crs4__cassandra)
.DocStr("Takes nothing returns something")
.NumInput(0)
.NumOutput(2)
.AddOptionalArg<std::vector<std::string>>("uuids", R"(A list of uuids)", nullptr)
.AddOptionalArg<std::vector<std::string>>("cass_ips", R"(List of Cassandra IPs)", nullptr)
.AddOptionalArg<std::vector<std::string>>("cass_conf", R"(Cassandra configuration parameters)", nullptr)
.AddOptionalArg("prefetch_queue_depth", R"(Number or prefetch buffers)", 1)
.AddOptionalArg("tcp_connections", R"(TCP connections used by Cassandra driver)", 2)
.AddOptionalArg("copy_threads", R"(Number of thread copying data in parallel)", 2);
