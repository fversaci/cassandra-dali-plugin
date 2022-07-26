#ifndef CRS4_CASSANDRA_H_
#define CRS4_CASSANDRA_H_

#include <vector>
#include <map>
#include <string>
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/reader/reader_op.h"
#include "batchhandler.h"
namespace crs4 {

class Cassandra : public ::dali::Operator<::dali::CPUBackend> {
public:
  explicit Cassandra(const ::dali::OpSpec &spec);

  // virtual inline ~Cassandra() = default;
  // Cassandra(const Cassandra&) = delete;
  // Cassandra& operator=(const Cassandra&) = delete;
  // Cassandra(Cassandra&&) = delete;
  // Cassandra& operator=(Cassandra&&) = delete;

  ::dali::ReaderMeta GetReaderMeta() const override {
    ::dali::ReaderMeta ret;
    ret.epoch_size = 10;
    ret.epoch_size_padded = 10;
    ret.number_of_shards = 1;
    ret.shard_id = 0;
    ret.pad_last_batch = true;
    ret.stick_to_shard = true;
    return ret;
  }

protected:
  bool CanInferOutputs() const override {
    return false;
  }

  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<::dali::CPUBackend> &ws) override {
    bh = new BatchHandler(cass_conf[0], cass_conf[1], cass_conf[2],
			  cass_conf[3], cass_conf[4], cass_conf[5],
			  cass_ips);
    return false;
  }

  void RunImpl(::dali::workspace_t<dali::CPUBackend> &ws) override;

private:
  std::vector<std::string> uuids;
  std::vector<std::string> cass_ips;
  std::vector<std::string> cass_conf;
  BatchHandler* bh;
};

}  // namespace crs4

#endif  // CRS4_CASSANDRA_H_
