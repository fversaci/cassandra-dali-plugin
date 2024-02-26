// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//
// Extension of https://github.com/NVIDIA/DALI/blob/main/dali/util/numpy.cc

#include "./dali_numpy.h"
#include "./numpy.h"

using namespace dali;

namespace crs4 {

  Tensor<CPUBackend> NewReadTensor(InputStream *src, bool pinned) {
  numpy::HeaderData header;
  numpy::ParseHeader(header, src);
  src->SeekRead(header.data_offset, SEEK_SET);

  Tensor<CPUBackend> data;
  data.set_pinned(pinned);
  data.Resize(header.shape, header.type());
  auto ret = src->Read(static_cast<uint8_t*>(data.raw_mutable_data()), header.nbytes());
  DALI_ENFORCE(ret == header.nbytes(), "Failed to read numpy file");

  if (header.fortran_order) {
    Tensor<CPUBackend> transposed;
    transposed.Resize(data.shape(), data.type());
    SampleView<CPUBackend> input(data.raw_mutable_data(), data.shape(), data.type());
    SampleView<CPUBackend> output(transposed.raw_mutable_data(), transposed.shape(),
                                  transposed.type());
    numpy::FromFortranOrder(output, input);
    return transposed;
  }
  return data;
}

}  // namespace crs4
