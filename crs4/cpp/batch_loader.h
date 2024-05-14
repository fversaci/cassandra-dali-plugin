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

enum lab_type {lab_int, lab_img, lab_none};

using INT_LABEL_T = int32_t;
using BatchRawImage = dali::TensorList<dali::CPUBackend>;
using BatchLabel = dali::TensorList<dali::CPUBackend>;
using BatchImgLab = std::pair<BatchRawImage, BatchLabel>;

class BatchLoader {
 private:
  // dali types
  dali::DALIDataType DALI_INT_TYPE = dali::DALI_INT32;
  dali::DALIDataType DALI_IMG_TYPE = dali::DALI_UINT8;
  // parameters
  bool connected = false;
  std::string table;
  lab_type label_t = lab_none;
  std::string label_col;
  std::string data_col;
  std::string id_col;
  std::string username;
  std::string password;
  std::vector<std::string> cassandra_ips;
  std::string cloud_config;
  std::string s_cassandra_ips;
  int port = 9042;
  bool use_ssl = false;
  std::string ssl_certificate;
  std::string ssl_own_certificate;
  std::string ssl_own_key;
  std::string ssl_own_key_pass;
  // Cassandra connection and execution
  CassCluster* cluster = cass_cluster_new();
  CassSession* session = cass_session_new();
  const CassPrepared* prepared;
  // concurrency
  ThreadPool* comm_pool;
  ThreadPool* copy_pool;
  ThreadPool* wait_pool;
  size_t io_threads;
  size_t copy_threads;  // copy parallelism
  size_t wait_threads;
  size_t comm_threads;  // number of communication threads
  size_t prefetch_buffers;  // multi-buffering
  bool ooo = false;  // enabling out-of-order?
  std::vector<std::mutex> alloc_mtx;
  std::vector<std::condition_variable> alloc_cv;
  std::vector<std::future<void>> comm_job;
  std::vector<std::vector<std::future<void>>> copy_jobs;
  // current batch
  std::vector<size_t> bs;
  std::vector<int> ooo_in_bs;  // how many images currently in ooo batches
  std::vector<std::future<BatchImgLab>> batch;
  std::vector<BatchRawImage> v_feats;
  std::vector<BatchLabel> v_labs;
  std::queue<int> read_buf;
  std::queue<int> write_buf;
  std::queue<int> ooo_buf;  // active ooo buffers
  std::mutex ooo_buf_mtx;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::vector<int64_t>> lab_shapes;
  // methods
  void connect();
  void check_connection();
  void copy_data_none(const CassResult* result, const cass_byte_t* data,
                      size_t sz, int off, int wb);
  void copy_data_int(const CassResult* result, const cass_byte_t* data,
                     size_t sz, cass_int32_t lab, int off, int wb);
  void copy_data_img(const CassResult* result, const cass_byte_t* data,
                     size_t sz, const cass_byte_t* lab, size_t l_sz,
                     int off, int wb);
  std::future<BatchImgLab> start_transfers(const std::vector<CassUuid>& keys,
                                           int wb);
  BatchImgLab wait4images(int wb);
  void keys2transfers(const std::vector<CassUuid>& keys, int wb);
  void transfer2copy(CassFuture* query_future, int wb, int i);
  void ooo_enqueue(CassFuture* query_future);
  static void wrap_enq(CassFuture* query_future, void* v_fd);
  void allocTens(int wb);
  void load_own_cert_file(std::string file, CassSsl* ssl);
  void load_own_key_file(std::string file, CassSsl* ssl, std::string passw);
  void load_trusted_cert_file(std::string file, CassSsl* ssl);
  void set_ssl(CassCluster* cluster);

 public:
  BatchLoader(std::string table, std::string label_type, std::string label_col,
              std::string data_col, std::string id_col, std::string username,
              std::string password, std::vector<std::string> cassandra_ips,
              int port, std::string cloud_config, bool use_ssl,
              std::string ssl_certificate, std::string ssl_own_certificate,
              std::string ssl_own_key, std::string ssl_own_key_pass,
              size_t io_threads, size_t prefetch_buffers, size_t copy_threads,
              size_t wait_threads, size_t comm_threads, bool ooo);
  ~BatchLoader();
  void prefetch_batch(const std::vector<CassUuid>& keys);
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
