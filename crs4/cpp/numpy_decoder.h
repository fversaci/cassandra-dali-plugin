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

#ifndef CRS4_CPP_NUMPY_DECODER_H_
#define CRS4_CPP_NUMPY_DECODER_H_

#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/util/numpy.h"

namespace crs4 {

class NumpyDecoder : public dali::Operator<dali::CPUBackend> {
 public:
  inline explicit NumpyDecoder(const dali::OpSpec &spec):
    dali::Operator<dali::CPUBackend>(spec) { }
  virtual inline ~NumpyDecoder() = default;

  NumpyDecoder(const NumpyDecoder&) = delete;
  NumpyDecoder& operator=(const NumpyDecoder&) = delete;
  NumpyDecoder(NumpyDecoder&&) = delete;
  NumpyDecoder& operator=(NumpyDecoder&&) = delete;

 protected:
  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                 const dali::Workspace &ws) override;
  void RunImpl(dali::Workspace &ws) override;
};

}  // namespace crs4

#endif  // CRS4_CPP_NUMPY_DECODER_H_
