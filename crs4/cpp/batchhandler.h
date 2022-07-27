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
#include "credentials.h"

#include "ThreadPool.h"

using RawImage = std::vector<char>;
using Label = int;
using BatchRawImage = std::vector<RawImage>;
using BatchLabel = std::vector<Label>;
using BatchImgLab = std::pair<BatchRawImage, BatchLabel>;

class BatchHandler{
private:
  // parameters
  bool connected = false;
  std::string table;
  std::string label_col;
  std::string data_col;
  std::string id_col;
  // std::vector<int> label_map;
  // bool use_label_map = false;
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
  ThreadPool* conv_pool;
  ThreadPool* batch_pool;
  int tcp_connections;
  int threads; // thread parallelism
  int batch_par;
  int comm_par; // number of communication threads
  int prefetch_buffers; // multi-buffering
  std::mutex glob_mtx;
  std::vector<std::mutex> alloc_mtx;
  std::vector<std::mutex> loc_mtx;
  std::vector<std::mutex> wait_mtx;
  std::vector<std::future<void>> comm_jobs;
  std::vector<std::vector<std::future<void>>> conv_jobs;
  // current batch
  std::vector<int> bs;
  std::vector<std::future<BatchImgLab>> batch;
  std::vector<BatchRawImage> v_feats;
  std::vector<BatchLabel> v_labs;
  std::vector<bool> allocated;
  std::queue<int> read_buf;
  std::queue<int> write_buf;
  // methods
  void connect();
  void check_connection();
  void img2tensor(const CassResult* result, int off, int wb);
  std::future<BatchImgLab> start_transfers(const std::vector<std::string>& keys, int wb);
  BatchImgLab wait4images(int wb);
  void keys2transfers(const std::vector<std::string>& keys, int wb);
  void transfer2conv(CassFuture* query_future, int wb, int i);
  static void wrap_t2c(CassFuture* query_future, void* v_fd);
  void allocTens(int wb);
public:
  BatchHandler(std::string table, std::string label_col, std::string data_col,
	       std::string id_col, // std::vector<int> label_map,
	       std::string username, std::string cass_pass,
	       std::vector<std::string> cassandra_ips, int tcp_connection=2,
	       int threads=2, int batch_par=2, int comm_par=1,
	       int prefetch_buffers=2, int port=9042);
  ~BatchHandler();
  void prefetch_batch_str(const std::vector<std::string>& keys);
  BatchImgLab blocking_get_batch();
  void ignore_batch();
};

struct futdata{
  BatchHandler* bh;
  int wb;
  int i;
};

#endif
