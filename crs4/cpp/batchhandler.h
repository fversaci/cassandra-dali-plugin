// Copyright 2021-2 CRS4
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef BATCHHANDLER_H
#define BATCHHANDLER_H

#include <cassandra.h>
#include <string>
#include <vector>
#include <future>
#include <utility>
#include <mutex>
#include "ThreadPool.h"
#include "dali/pipeline/operator/operator.h"
#include "credentials.h"

using RawImage = std::vector<char>;
using Label = int32_t;
// using BatchRawImage = std::vector<RawImage>;
using BatchRawImage = ::dali::TensorVector<::dali::CPUBackend>;
// using TensImage = ::dali::Tensor<::dali::CPUBackend>;
using BatchLabel = ::dali::TensorVector<::dali::CPUBackend>;
using BatchImgLab = std::pair<BatchRawImage, BatchLabel>;

class BatchHandler {
private:
  // parameters
  bool connected = false;
  std::string table;
  std::string label_col;
  std::string data_col;
  std::string id_col;
  std::string username;
  std::string password;
  std::vector<std::string> cassandra_ips;
  std::string s_cassandra_ips;
  int port = 9042;
  // Cassandra connection and execution
  CassCluster* cluster = cass_cluster_new();
  CassSession* session = cass_session_new();
  const CassPrepared* prepared;
  // concurrency
  ThreadPool* comm_pool;
  ThreadPool* copy_pool;
  ThreadPool* wait_pool;
  int tcp_connections;
  int copy_threads; // copy parallelism
  int wait_par;
  int comm_par; // number of communication threads
  int prefetch_buffers; // multi-buffering
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
  std::vector<int> shape_count;
  // methods
  void connect();
  void check_connection();
  void img2tensor(const CassResult* result, const cass_byte_t* data,
                  size_t sz, cass_int32_t lab, int off, int wb);
  std::future<BatchImgLab> start_transfers(const std::vector<std::string>& keys, int wb);
  BatchImgLab wait4images(int wb);
  void keys2transfers(const std::vector<std::string>& keys, int wb);
  void transfer2conv(CassFuture* query_future, int wb, int i);
  static void wrap_t2c(CassFuture* query_future, void* v_fd);
  void allocTens(int wb);
public:
  BatchHandler(std::string table, std::string label_col, std::string data_col,
               std::string id_col,
               std::string username, std::string cass_pass,
               std::vector<std::string> cassandra_ips, int prefetch_buffers,
               int tcp_connections, int copy_threads, int wait_par=2,
               int comm_par=2, int port=9042);
  ~BatchHandler();
  void prefetch_batch(const std::vector<std::string>& keys);
  BatchImgLab blocking_get_batch();
  void ignore_batch();
};

struct futdata {
  BatchHandler* bh;
  int wb;
  int i;
};

#endif
