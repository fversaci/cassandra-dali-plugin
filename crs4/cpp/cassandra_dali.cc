// Copyright 2021-2 CRS4
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <iostream>
#include <fstream>

#include "./cassandra_dali.h"

namespace crs4 {

Cassandra::Cassandra(const ::dali::OpSpec &spec) :
  ::dali::InputOperator<dali::CPUBackend>(spec),
  batch_size(spec.GetArgument<int>("max_batch_size")),
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
  prefetch_buffers(spec.GetArgument<int>("prefetch_buffers")),
  io_threads(spec.GetArgument<int>("io_threads")),
  copy_threads(spec.GetArgument<int>("copy_threads")),
  wait_threads(spec.GetArgument<int>("wait_threads")),
  comm_threads(spec.GetArgument<int>("comm_threads")),
  ooo(spec.GetArgument<bool>("ooo")) {
  DALI_ENFORCE(label_type == "int" || label_type == "image" || label_type == "none",
               "label_type can only be int, image or none.");
  batch_ldr = new BatchLoader(table, label_type, label_col, data_col, id_col,
                        username, password, cassandra_ips, cassandra_port,
                        cloud_config, use_ssl, ssl_certificate,
                        io_threads, prefetch_buffers, copy_threads,
                        wait_threads, comm_threads, ooo);
}

void Cassandra::prefetch_one(const dali::TensorList<dali::CPUBackend>& input) {
  assert(batch_size == input.num_samples());
  auto cass_uuids = std::vector<CassUuid>(batch_size);
  for (auto i=0; i != batch_size; ++i) {
    auto d_ptr = input[i].data<dali::uint64>();
    auto c_uuid = &cass_uuids[i];
    c_uuid->time_and_version = *(d_ptr++);
    c_uuid->clock_seq_and_node = *d_ptr;
  }
  batch_ldr->prefetch_batch(cass_uuids);
}

bool Cassandra::SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                          const ::dali::Workspace &ws) {
  // link input data to uuids tensorlist
  InputOperator<::dali::CPUBackend>::HandleDataAvailability();
  uuids.Reset();
  uuids.set_pinned(false);
  auto &thread_pool = ws.GetThreadPool();
  ForwardCurrentData(uuids, null_data_id, thread_pool);

  return false;
}

void Cassandra::fill_buffers(::dali::Workspace &ws) {
  // start prefetching
  for (int i=0; i < prefetch_buffers; ++i) {
    prefetch_one(uuids);
    auto &thread_pool = ws.GetThreadPool();
    ForwardCurrentData(uuids, null_data_id, thread_pool);
  }
  buffers_empty = false;
}

void Cassandra::RunImpl(::dali::Workspace &ws) {
  if (buffers_empty) {
    fill_buffers(ws);
  }
  BatchImgLab batch = batch_ldr->blocking_get_batch();
  // std::cout << (int) *(batch.second[0].data<dali::int32>()) << std::endl;
  prefetch_one(uuids);
  // share features with output
  auto &features = ws.Output<::dali::CPUBackend>(0);
  features.ShareData(batch.first);
  // share labels with output
  auto &labels = ws.Output<::dali::CPUBackend>(1);
  labels.ShareData(batch.second);
}

}  // namespace crs4

DALI_REGISTER_OPERATOR(crs4__cassandra, ::crs4::Cassandra, ::dali::CPU);

DALI_SCHEMA(crs4__cassandra)
.DocStr("Reads UUIDs via feed_pipeline and returns images and labels/masks")
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
   R"(Optional SSL certificate)", "")
.AddOptionalArg("prefetch_buffers", R"(Number or prefetch buffers)", 1)
.AddOptionalArg("io_threads",
   R"(Number of io threads used by the Cassandra driver)", 2)
.AddOptionalArg("copy_threads",
   R"(Number of threads copying data in parallel)", 2)
.AddOptionalArg("wait_threads", R"(Parallelism for waiting threads)", 2)
.AddOptionalArg("comm_threads", R"(Parallelism for communication threads)", 2)
.AddOptionalArg("blocking", R"(block until the data is available)", true)
.AddOptionalArg("no_copy", R"(should DALI copy the buffer when ``feed_input`` is called?)", false)
.AddOptionalArg("ooo", R"(Enable out-of-order batches)", false)
.AddParent("InputOperatorBase");
