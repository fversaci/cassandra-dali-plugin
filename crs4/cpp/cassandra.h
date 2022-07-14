#ifndef CRS4_CASSANDRA_H_
#define CRS4_CASSANDRA_H_

#include <vector>
#include "dali/pipeline/operator/operator.h"

namespace crs4 {

template <typename Backend>
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
        return true;
    }

    bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                   const ::dali::workspace_t<Backend> &ws) override {
        const auto &input = ws.template Input<Backend>(0);
        output_desc.resize(1);
        output_desc[0] = {input.shape(), input.type()};
        return true;
    }

    void RunImpl(::dali::workspace_t<Backend> &ws) override;
};

}  // namespace crs4

#endif  // CRS4_CASSANDRA_H_
