// Copyright 2021-2 CRS4
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef CRS4_CPP_BATCH_LOADER_H_
#define CRS4_CPP_BATCH_LOADER_H_

#include <cassandra.h>
#include <string>
#include <queue>
#include <vector>
#include <future>
#include <utility>
#include <mutex>
#include "dali/pipeline/operator/operator.h"
#include "ThreadPool.h"

namespace crs4 {

using LABEL_TYPE = int32_t;
using BatchRawImage = ::dali::TensorList<::dali::CPUBackend>;
using BatchLabel = ::dali::TensorList<::dali::CPUBackend>;
using BatchImgLab = std::pair<BatchRawImage, BatchLabel>;

class BatchLoader {
 private:
  // dali types
  dali::DALIDataType DALI_LABEL_TYPE = ::dali::DALI_INT32;
  dali::DALIDataType DALI_FEAT_TYPE = ::dali::DALI_UINT8;
  // parameters
  bool connected = false;
  std::string table;
  std::string label_col;
  std::string data_col;
  std::string id_col;
  std::string username;
  std::string password;
  std::string cloud_config;
  std::vector<std::string> cassandra_ips;
  std::string s_cassandra_ips;
  int port = 9042;
  bool use_ssl = false;
  std::string ssl_certificate;
  // Cassandra connection and execution
  CassCluster* cluster = cass_cluster_new();
  CassSession* session = cass_session_new();
  const CassPrepared* prepared;
  // concurrency
  ThreadPool* comm_pool;
  ThreadPool* copy_pool;
  ThreadPool* wait_pool;
  int io_threads;
  int copy_threads;  // copy parallelism
  int wait_threads;
  int comm_threads;  // number of communication threads
  int prefetch_buffers;  // multi-buffering
  std::vector<std::mutex> alloc_mtx;
  std::vector<std::condition_variable> alloc_cv;
  std::vector<std::future<void>> comm_jobs;
  std::vector<std::vector<std::future<void>>> copy_jobs;
  // current batch
  std::vector<int> bs;
  std::vector<std::future<BatchImgLab>> batch;
  std::vector<BatchRawImage> v_feats;
  std::vector<BatchLabel> v_labs;
  std::queue<int> read_buf;
  std::queue<int> write_buf;
  std::vector<std::vector<int64_t>> shapes;
  // methods
  void connect();
  void check_connection();
  void copy_data(const CassResult* result, const cass_byte_t* data,
                  size_t sz, cass_int32_t lab, int off, int wb);
  std::future<BatchImgLab> start_transfers(const std::vector<std::string>& keys,
                                           int wb);
  BatchImgLab wait4images(int wb);
  void keys2transfers(const std::vector<std::string>& keys, int wb);
  void transfer2copy(CassFuture* query_future, int wb, int i);
  static void wrap_t2c(CassFuture* query_future, void* v_fd);
  void allocTens(int wb);

 public:
  BatchLoader(std::string table, std::string label_col, std::string data_col,
              std::string id_col, std::string username, std::string password,
              std::vector<std::string> cassandra_ips, int port,
              std::string cloud_config, bool use_ssl,
              std::string ssl_certificate, int io_threads,
              int prefetch_buffers, int copy_threads,
              int wait_threads, int comm_threads);
  ~BatchLoader();
  void prefetch_batch(const std::vector<std::string>& keys);
  BatchImgLab blocking_get_batch();
  void ignore_batch();
};

struct futdata {
  BatchLoader* batch_ldr;
  int wb;
  int i;
};

}  // namespace crs4

#endif  // CRS4_CPP_BATCH_LOADER_H_
