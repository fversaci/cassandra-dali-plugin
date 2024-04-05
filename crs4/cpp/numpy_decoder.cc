// Copyright 2024 CRS4 (http://www.crs4.it/)
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
#include "./numpy_decoder.h"

namespace crs4 {

bool NumpyDecoder::SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                             const dali::Workspace &ws) {
  return false;
}

void NumpyDecoder::RunImpl(dali::Workspace &ws) {
  const auto &input = ws.Input<::dali::CPUBackend>(0);
  auto &output = ws.Output<::dali::CPUBackend>(0);
  const auto &in_shape = input.shape();
  bool pinned = output.is_pinned();
  for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
    auto in_ptr = input.raw_tensor(sample_id);
    auto sz = in_shape.tensor_size(sample_id);
    auto mis = ::dali::MemInputStream(in_ptr, sz);
    auto npy_ten = ReadTensor(&mis, pinned);
    output.set_type(npy_ten.type());
    output.CopySample(sample_id, npy_ten);
  }
}

}  // namespace crs4

// register NumpyDecoder class

DALI_REGISTER_OPERATOR(crs4__numpy_decoder, crs4::NumpyDecoder, dali::CPU);

DALI_SCHEMA(crs4__numpy_decoder)
.DocStr("Decodes NumPy .npy files, that have already been loaded into memory, into tensors.")
.NumInput(1)
.NumOutput(1);

