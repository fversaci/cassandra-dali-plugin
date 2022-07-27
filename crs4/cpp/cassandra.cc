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
  bs(spec.GetArgument<int>("max_batch_size")),
  uuids(spec.GetArgument<std::vector<std::string>>("uuids")),
  cass_ips(spec.GetArgument<std::vector<std::string>>("cass_ips")),
  cass_conf(spec.GetArgument<std::vector<std::string>>("cass_conf"))
{
  bh = new BatchHandler(cass_conf[0], cass_conf[1], cass_conf[2],
			cass_conf[3], cass_conf[4], cass_conf[5],
			cass_ips);
  auto batch_ids = std::vector(uuids.begin(), uuids.begin()+bs);
  bh->prefetch_batch_str(batch_ids);
  std::cout << batch_ids[0] << std::endl;
}


void Cassandra::RunImpl(::dali::HostWorkspace &ws) {
  // // read test image
  // std::ifstream is ("coso.jpg", std::ifstream::binary);
  // is.seekg (0, is.end);
  // int length = is.tellg();
  // is.seekg (0, is.beg);
  // char* buffer = new char[length];
  // // std::cout << "Reading " << length << " characters... " << std::endl;
  // is.read (buffer,length);

  auto keys = std::vector(uuids.begin(), uuids.begin()+bs);
  BatchRawImage bri = bh->blocking_get_batch().first;
  auto batch_ids = std::vector(uuids.begin(), uuids.begin()+bs);
  bh->prefetch_batch_str(batch_ids);
  int ibs = bri.size(); // input batch size
  std::vector<int64_t> sz;
  for (auto i = 0; i != ibs; ++i){
    sz.push_back(bri[i].size());
  }
  ::dali::TensorListShape szs(sz, ibs, 1);
  
  // copy to output
  auto &output = ws.Output<::dali::CPUBackend>(0);
  // output.SetContiguous(true);
  output.Resize(szs, ::dali::DALI_UINT8);
  // std::cout << output.IsContiguous() << std::endl;

  auto &tp = ws.GetThreadPool();
  for (int i = 0; i < ibs; ++i) {
    // std::cout << output.tensor_shape(sample_id) << std::endl;
    tp.AddWork(
    [&, i](int thread_id) {
      std::memcpy(
        output.raw_mutable_tensor(i),
        bri[i].data(),
        bri[i].size());
    });
  }
  tp.RunAll();
}

}  // namespace crs4

DALI_REGISTER_OPERATOR(crs4__cassandra,
                       ::crs4::Cassandra, ::dali::CPU);

DALI_SCHEMA(crs4__cassandra)
.DocStr("Takes nothing returns something")
.NumInput(0)
.NumOutput(1)
.AddOptionalArg<std::vector<std::string>>("uuids", R"(A list of uuids)", nullptr)
.AddOptionalArg<std::vector<std::string>>("cass_ips", R"(List of Cassandra IPs)", nullptr)
.AddOptionalArg<std::vector<std::string>>("cass_conf", R"(Cassandra configuration parameters)", nullptr);
