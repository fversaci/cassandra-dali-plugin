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
  cassandra_ips(spec.GetArgument<std::vector<std::string>>("cassandra_ips")),
  cassandra_port(spec.GetArgument<int>("cassandra_port")),
  table(spec.GetArgument<std::string>("table")),
  label_col(spec.GetArgument<std::string>("label_col")),
  data_col(spec.GetArgument<std::string>("data_col")),
  id_col(spec.GetArgument<std::string>("id_col")),
  username(spec.GetArgument<std::string>("username")),
  password(spec.GetArgument<std::string>("password")),
  use_ssl(spec.GetArgument<bool>("use_ssl")),
  ssl_certificate(spec.GetArgument<std::string>("ssl_certificate")),
  shuffle_after_epoch(spec.GetArgument<bool>("shuffle_after_epoch")),
  prefetch_buffers(spec.GetArgument<int>("prefetch_buffers")),
  io_threads(spec.GetArgument<int>("io_threads")),
  copy_threads(spec.GetArgument<int>("copy_threads")),
  wait_threads(spec.GetArgument<int>("wait_threads")),
  comm_threads(spec.GetArgument<int>("comm_threads"))
{
  bh = new BatchHandler(table, label_col, data_col, id_col,
			username, password, cassandra_ips, cassandra_port,
			use_ssl, ssl_certificate,
			io_threads, prefetch_buffers, copy_threads,
			wait_threads, comm_threads);
  Reset();
  // start prefetching
  for (int i=0; i<prefetch_buffers; ++i) {
    prefetch_one();
  }
}

void Cassandra::prefetch_one() {
  auto dist = std::distance(current, uuids.end());
  // if reached the end, rewind
  if (dist==0) {
    Reset();
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
  features.ShareData(batch.first);
  // copy labels to output
  auto &labels = ws.Output<::dali::CPUBackend>(1);
  labels.ShareData(batch.second);
}

}  // namespace crs4

DALI_REGISTER_OPERATOR(crs4__cassandra, ::crs4::Cassandra, ::dali::CPU);

DALI_SCHEMA(crs4__cassandra)
.DocStr("Takes nothing returns something")
.NumInput(0)
.NumOutput(2)
.AddOptionalArg<std::vector<std::string>>("uuids",
   R"(A list of uuids)", nullptr)
.AddOptionalArg<std::vector<std::string>>("cassandra_ips",
   R"(List of Cassandra IPs)", nullptr)
.AddOptionalArg("cassandra_port",
   R"(Port to connect to in the Cassandra server)", 9042)
.AddOptionalArg<std::string>("table", R"()", nullptr)
.AddOptionalArg<std::string>("label_col", R"()", nullptr)
.AddOptionalArg<std::string>("data_col", R"()", nullptr)
.AddOptionalArg<std::string>("id_col", R"()", nullptr)
.AddOptionalArg<std::string>("username", R"()", nullptr)
.AddOptionalArg<std::string>("password", R"()", nullptr)
.AddOptionalArg("use_ssl", R"(Encrypt Cassandra connection with SSL)", false)
.AddOptionalArg<std::string>("ssl_certificate",
			     R"(Optional SSL certificate)", "")
.AddOptionalArg("shuffle_after_epoch", R"(Reshuffling uuids at each epoch)",
		false)
.AddOptionalArg("prefetch_buffers", R"(Number or prefetch buffers)", 1)
.AddOptionalArg("io_threads",
   R"(Number of io threads used by the Cassandra driver)", 2)
.AddOptionalArg("copy_threads",
   R"(Number of threads copying data in parallel)", 2)
.AddOptionalArg("wait_threads", R"(Parallelism for waiting threads)", 2)
.AddOptionalArg("comm_threads", R"(Parallelism for communication threads)", 2);
