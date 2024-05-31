// Copyright 2022 CRS4 (http://www.crs4.it/)
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

#ifndef CRS4_CPP_CASSANDRA_DALI_DECOUPLED_H_
#define CRS4_CPP_CASSANDRA_DALI_DECOUPLED_H_

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cmath>
#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/operators/reader/reader_op.h"
#include "./cassandra_dali_interactive.h"
#include "./batch_loader.h"

namespace crs4 {

class CassandraDecoupled : public CassandraInteractive {
 public:
  explicit CassandraDecoupled(const dali::OpSpec &spec);

 protected:
  bool SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                 const dali::Workspace &ws) override;
  void RunImpl(dali::Workspace &ws) override;

 private:
  int mini_batch_size;
  void prefetch_one();
  void list_to_minibatches(const dali::Workspace &ws);
  void fill_buffers(dali::Workspace &ws);
  std::vector<std::pair<size_t, size_t>> intervals;
  size_t input_interval = 0;
  BatchImgLab output;
};

}  // namespace crs4

#endif  // CRS4_CPP_CASSANDRA_DALI_DECOUPLED_H_
