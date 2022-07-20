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
  std::cout << output.num_samples() << std::endl;
  image.Resize({length}, ::dali::DALI_UINT8);
  std::memcpy(image.raw_mutable_data(), buffer, length);
  std::cout << "Pin: " << output.is_pinned() << std::endl;

  ::dali::TypeInfo type = image.type_info();
  auto &tp = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < output.num_samples(); sample_id++) {
    std::cout << output.tensor_shape(sample_id) << std::endl;
    tp.AddWork(
    [&, sample_id](int thread_id) {
      type.Copy<::dali::CPUBackend, ::dali::CPUBackend>(
        output.raw_mutable_tensor(sample_id),
        image.raw_mutable_data(),
        image.size(), 0);
    },
    image.size());
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
