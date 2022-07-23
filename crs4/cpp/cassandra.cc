#include <iostream>
#include <fstream>

#include "cassandra.h"

namespace crs4 {


Cassandra::Cassandra(const ::dali::OpSpec &spec) :
  ::dali::Operator<dali::CPUBackend>(spec),
  uuids(spec.GetArgument<std::vector<std::string>>("uuids"))
{
  std::cout << uuids[0] << std::endl;
}


void Cassandra::RunImpl(::dali::HostWorkspace &ws) {
  // read test image
  std::ifstream is ("coso.jpg", std::ifstream::binary);
  is.seekg (0, is.end);
  int length = is.tellg();
  is.seekg (0, is.beg);
  char* buffer = new char[length];
  // std::cout << "Reading " << length << " characters... " << std::endl;
  is.read (buffer,length);

  // copy to output
  auto &output = ws.Output<::dali::CPUBackend>(0);
  int bs = output.num_samples();
  // output.SetContiguous(true);

  std::vector<int64_t> sz(bs, length);
  ::dali::TensorListShape szs(sz, bs, 1);
  // std::cout << szs << std::endl;

  output.Resize(szs, ::dali::DALI_UINT8);
  // std::cout << output.IsContiguous() << std::endl;

  auto &tp = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < bs; sample_id++) {
    // std::cout << output.tensor_shape(sample_id) << std::endl;
    tp.AddWork(
    [&, sample_id, buffer, length](int thread_id) {
      std::memcpy(
        output.raw_mutable_tensor(sample_id),
        buffer,
        length);
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
.AddOptionalArg<std::vector<std::string>>("uuids", R"(A list of uuids)", nullptr);
