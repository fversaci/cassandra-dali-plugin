#include <iostream>
#include <fstream>

#include "cassandra.h"

namespace crs4 {

template <>
void Cassandra<::dali::CPUBackend>::RunImpl(::dali::HostWorkspace &ws) {
  // testing stuff
  std::ifstream is ("coso.jpg", std::ifstream::binary);
  is.seekg (0, is.end);
  int length = is.tellg();
  is.seekg (0, is.beg);
  char* buffer = new char[length];
  std::cout << "Reading " << length << " characters... ";
  is.read (buffer,length);
  
  // original code
  const auto &input = ws.Input<::dali::CPUBackend>(0);
  auto &output = ws.Output<::dali::CPUBackend>(0);

  ::dali::TypeInfo type = input.type_info();
  auto &tp = ws.GetThreadPool();
  const auto &in_shape = input.shape();
  for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
    tp.AddWork(
    [&, sample_id](int thread_id) {
      type.Copy<::dali::CPUBackend, ::dali::CPUBackend>(
        output.raw_mutable_tensor(sample_id),
        input.raw_tensor(sample_id),
        in_shape.tensor_size(sample_id), 0);
    },
    in_shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

}  // namespace crs4

DALI_REGISTER_OPERATOR(Crs4Cassandra,
                       ::crs4::Cassandra<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(Crs4Cassandra)
.DocStr("Make a copy of the input tensor")
.NumInput(1)
.NumOutput(1);
