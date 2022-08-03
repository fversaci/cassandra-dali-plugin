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
  cass_conf(spec.GetArgument<std::vector<std::string>>("cass_conf"))
{
  bh = new BatchHandler(cass_conf[0], cass_conf[1], cass_conf[2],
			cass_conf[3], cass_conf[4], cass_conf[5],
			cass_ips);
  auto batch_ids = std::vector(uuids.begin(), uuids.begin()+batch_size);
  bh->prefetch_batch(batch_ids);
}

void Cassandra::RunImpl(::dali::HostWorkspace &ws) {
  auto keys = std::vector(uuids.begin(), uuids.begin()+batch_size);
  BatchImgLab batch = bh->blocking_get_batch();
  BatchRawImage feats = batch.first;
  auto batch_ids = std::vector(uuids.begin(), uuids.begin()+batch_size);
  bh->prefetch_batch(batch_ids);
  int ibs = feats.size(); // input batch size
  std::vector<int64_t> sz;
  for (auto i = 0; i != ibs; ++i){
    sz.push_back(feats[i].size());
  }
  ::dali::TensorListShape szs(sz, ibs, 1);
  
  // copy features to output
  auto &features = ws.Output<::dali::CPUBackend>(0);
  // features.SetContiguous(true);
  features.Resize(szs, ::dali::DALI_UINT8);
  // std::cout << features.IsContiguous() << std::endl;

  auto &tp = ws.GetThreadPool();
  for (int i = 0; i < ibs; ++i) {
    // std::cout << features.tensor_shape(sample_id) << std::endl;
    tp.AddWork(
    [&, i](int thread_id) {
      std::memcpy(
        features.raw_mutable_tensor(i),
        feats[i].data(),
        feats[i].size());
    });
  }
  tp.RunAll();

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
.AddOptionalArg<std::vector<std::string>>("cass_conf", R"(Cassandra configuration parameters)", nullptr);
