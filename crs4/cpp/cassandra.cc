#include <iostream>
#include <fstream>

#include "cassandra.h"

namespace crs4 {

template <>
void Cassandra<::dali::CPUBackend>::RunImpl(::dali::HostWorkspace &ws) {
  // read test image
  std::ifstream is ("coso.jpg", std::ifstream::binary);
  is.seekg (0, is.end);
  int length = is.tellg();
  is.seekg (0, is.beg);
  char* buffer = new char[length];
  std::cout << "Reading " << length << " characters... " << std::endl;
  is.read (buffer,length);

  // copy to output
  auto &output = ws.Output<::dali::CPUBackend>(0);
  // output.SetContiguous(true);
  output.Resize({{14409}, {14409}}, ::dali::DALI_UINT8);
  // std::cout << output.IsContiguous() << std::endl;

  auto &tp = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < output.num_samples(); sample_id++) {
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

DALI_REGISTER_OPERATOR(Crs4Cassandra,
                       ::crs4::Cassandra<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(Crs4Cassandra)
.DocStr("Takes nothing returns something")
.NumInput(0)
.NumOutput(1);
