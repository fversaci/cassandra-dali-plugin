#ifndef CRS4_CASSANDRA_H_
#define CRS4_CASSANDRA_H_

#include <vector>
#include "dali/pipeline/operator/operator.h"

namespace crs4 {

template <typename Backend = ::dali::CPUBackend>
class Cassandra : public ::dali::Operator<Backend> {
public:
  inline explicit Cassandra(const ::dali::OpSpec &spec) :
    ::dali::Operator<Backend>(spec) {}

  virtual inline ~Cassandra() = default;

  Cassandra(const Cassandra&) = delete;
  Cassandra& operator=(const Cassandra&) = delete;
  Cassandra(Cassandra&&) = delete;
  Cassandra& operator=(Cassandra&&) = delete;

protected:
  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<Backend> &ws) override {

    // output_desc.resize(1);
    // output_desc[0] = {{{14409}, {14409}}, ::dali::DALI_UINT8};
    return false;
  }

  void RunImpl(::dali::workspace_t<Backend> &ws) override;
};

}  // namespace crs4

#endif  // CRS4_CASSANDRA_H_
