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

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include "./batch_loader.h"

namespace crs4 {

BatchLoader::~BatchLoader() {
  if (connected) {
    ignore_batch();
    cass_session_free(session);
    cass_cluster_free(cluster);
    delete(copy_pool);
    delete(comm_pool);
    delete(wait_pool);
  }
}

void BatchLoader::load_own_cert_file(std::string file, CassSsl* ssl) {
  CassError rc;
  char* cert;
  int64_t cert_size;

  FILE *in = fopen(file.c_str(), "rb");
  if (in == NULL) {
    throw std::runtime_error("Error loading certificate file " + file);
  }

  fseek(in, 0, SEEK_END);
  cert_size = ftell(in);
  rewind(in);

  cert = reinterpret_cast<char*>(malloc(cert_size));
  fread(cert, sizeof(char), cert_size, in);
  fclose(in);

  // Add the trusted certificate (or chain) to the driver
  rc = cass_ssl_set_cert_n(ssl, cert, cert_size);
  if (rc != CASS_OK) {
    free(cert);
    throw std::runtime_error("Error loading SSL certificate: "
                             + std::string(cass_error_desc(rc)));
  }

  free(cert);
}

void BatchLoader::load_own_key_file(std::string file, CassSsl* ssl, std::string passw) {
  CassError rc;
  char* cert;
  int64_t cert_size;

  // Get the C-style password
  const char* password = passw.c_str();
  size_t password_length = passw.length();

  FILE *in = fopen(file.c_str(), "rb");
  if (in == NULL) {
    throw std::runtime_error("Error loading certificate file " + file);
  }

  fseek(in, 0, SEEK_END);
  cert_size = ftell(in);
  rewind(in);

  cert = reinterpret_cast<char*>(malloc(cert_size));
  fread(cert, sizeof(char), cert_size, in);
  fclose(in);

  // Add the trusted certificate (or chain) to the driver
  rc = cass_ssl_set_private_key_n(ssl, cert, cert_size, password, password_length);
  if (rc != CASS_OK) {
    free(cert);
    throw std::runtime_error("Error loading SSL certificate: "
                             + std::string(cass_error_desc(rc)));
  }

  free(cert);
}

void BatchLoader::load_trusted_cert_file(std::string file, CassSsl* ssl) {
  CassError rc;
  char* cert;
  int64_t cert_size;

  FILE *in = fopen(file.c_str(), "rb");
  if (in == NULL) {
    throw std::runtime_error("Error loading certificate file " + file);
  }

  fseek(in, 0, SEEK_END);
  cert_size = ftell(in);
  rewind(in);

  cert = reinterpret_cast<char*>(malloc(cert_size));
  fread(cert, sizeof(char), cert_size, in);
  fclose(in);

  // Add the trusted certificate (or chain) to the driver
  rc = cass_ssl_add_trusted_cert_n(ssl, cert, cert_size);
  if (rc != CASS_OK) {
    free(cert);
    throw std::runtime_error("Error loading SSL certificate: "
                             + std::string(cass_error_desc(rc)));
  }

  free(cert);
}

void BatchLoader::set_ssl(CassCluster* cluster) {
  CassSsl* ssl = cass_ssl_new();
  // verify server certificate
  if (ssl_certificate.empty()) {
    cass_ssl_set_verify_flags(ssl, CASS_SSL_VERIFY_NONE);
  } else {
    load_trusted_cert_file(ssl_certificate, ssl);
  }
  // client authentication, needs both certificate and key
  if (!ssl_own_certificate.empty() && !ssl_own_key.empty()) {
    load_own_cert_file(ssl_own_certificate, ssl);
    load_own_key_file(ssl_own_key, ssl, ssl_own_key_pass);
  }
  cass_cluster_set_ssl(cluster, ssl);
  cass_ssl_free(ssl);
}

void BatchLoader::connect() {
  if (cloud_config.empty()) {
    // direct connection
    cass_cluster_set_contact_points(cluster, s_cassandra_ips.c_str());
    cass_cluster_set_port(cluster, port);
  } else {
    // cloud configuration (e.g., AstraDB)
    if (cass_cluster_set_cloud_secure_connection_bundle(cluster,
                                                        cloud_config.c_str())
        != CASS_OK) {
      throw std::runtime_error(
           "Unable to configure cloud using the secure connection bundle: "
           + cloud_config);
    }
  }
  cass_cluster_set_connect_timeout(cluster, 10000);
  cass_cluster_set_request_timeout(cluster, 60000);
  cass_cluster_set_credentials(cluster, username.c_str(), password.c_str());
  cass_cluster_set_protocol_version(cluster, CASS_PROTOCOL_VERSION_V4);
  cass_cluster_set_num_threads_io(cluster, io_threads);
  cass_cluster_set_application_name(cluster,
                               "Cassandra module for NVIDIA DALI, CRS4");
  cass_cluster_set_queue_size_io(cluster, 65536);  // max pending requests
  // set ssl if required
  if (use_ssl) {
    set_ssl(cluster);
  }
  CassFuture* connect_future = cass_session_connect(session, cluster);
  CassError rc = cass_future_error_code(connect_future);
  cass_future_free(connect_future);
  if (rc != CASS_OK) {
    throw std::runtime_error("Error: unable to connect to Cassandra DB. ");
  }
  // assemble query
  std::stringstream ss;
  ss << "SELECT ";
  if (label_t != lab_none) {  // getting label/mask?
    ss << label_col << ", ";
  }
  ss << data_col << " FROM " << table << " WHERE "
     << id_col << "=?" << std::endl;
  std::string query = ss.str();
  // prepare statement
  CassFuture* prepare_future = cass_session_prepare(session, query.c_str());
  prepared = cass_future_get_prepared(prepare_future);
  if (prepared == NULL) {
    /* Handle error */
    cass_future_free(prepare_future);
    throw std::runtime_error("Error in query: " + query);
  }
  cass_future_free(prepare_future);
  // init thread pools
  comm_pool = new ThreadPool(comm_threads);
  copy_pool = new ThreadPool(copy_threads);
  wait_pool = new ThreadPool(wait_threads);
}

BatchLoader::BatchLoader(std::string table, std::string label_type,
                         std::string label_col, std::string data_col,
                         std::string id_col, std::string username, std::string password,
                         std::vector<std::string> cassandra_ips, int port,
                         std::string cloud_config, bool use_ssl,
                         std::string ssl_certificate, std::string ssl_own_certificate,
                         std::string ssl_own_key, std::string ssl_own_key_pass,
                         size_t io_threads, size_t prefetch_buffers, size_t copy_threads,
                         size_t wait_threads, size_t comm_threads, bool ooo) :
  table(table), label_col(label_col), data_col(data_col), id_col(id_col),
  username(username), password(password), cassandra_ips(cassandra_ips),
  cloud_config(cloud_config), port(port), use_ssl(use_ssl),
  ssl_certificate(ssl_certificate), ssl_own_certificate(ssl_own_certificate),
  ssl_own_key(ssl_own_key), ssl_own_key_pass(ssl_own_key_pass),
  io_threads(io_threads), copy_threads(copy_threads),
  wait_threads(wait_threads), comm_threads(comm_threads),
  prefetch_buffers(prefetch_buffers), ooo(ooo) {
  // setting label type, default is lab_none
  if (label_type == "int") {
    label_t = lab_int;
  } else if (label_type == "blob") {
    label_t = lab_img;
    lab_shapes.resize(prefetch_buffers);
  }
  // init multi-buffering variables
  bs.resize(prefetch_buffers);
  ooo_in_bs.resize(prefetch_buffers);
  batch.resize(prefetch_buffers);
  copy_jobs.resize(prefetch_buffers);
  comm_job.resize(prefetch_buffers);
  v_feats.resize(prefetch_buffers);
  v_labs.resize(prefetch_buffers);
  shapes.resize(prefetch_buffers);
  alloc_cv = std::vector<std::condition_variable>(prefetch_buffers);
  alloc_mtx = std::vector<std::mutex>(prefetch_buffers);
  for (size_t i = 0; i < prefetch_buffers; ++i) {
    write_buf.push(i);
  }
  // join cassandra ip's into comma seperated string
  s_cassandra_ips =
    std::accumulate(cassandra_ips.begin(), cassandra_ips.end(),
                    std::string(),
  [](const std::string& a, const std::string& b) -> std::string {
    return a + (a.length() > 0 ? "," : "") + b;
  });
}

void BatchLoader::allocTens(int wb) {
  shapes[wb].clear();
  shapes[wb].resize(bs[wb]);
  v_feats[wb] = BatchRawImage();
  v_feats[wb].set_pinned(false);
  // v_feats[wb].SetContiguous(true);
  v_labs[wb] = BatchLabel();
  v_labs[wb].set_pinned(false);
  // v_labs[wb].SetContiguous(true);
  if (label_t == lab_img) {
    lab_shapes[wb].clear();
    lab_shapes[wb].resize(bs[wb]);
  } else {
    // if labels are not images we can already allocate the memory
    std::vector<int64_t> v_sz(bs[wb], 1);
    dali::TensorListShape t_sz(v_sz, bs[wb], 1);
    v_labs[wb].Resize(t_sz, DALI_INT_TYPE);
  }
}

void BatchLoader::copy_data_none(const CassResult* result,
                                 const cass_byte_t* data, size_t sz,
                                 int off, int wb) {
  // wait for feature tensor to be allocated
  {
    std::unique_lock<std::mutex> lck(alloc_mtx[wb]);
    while (copy_jobs[wb].size() != bs[wb]) {
      alloc_cv[wb].wait(lck);
    }
  }
  // copy data in batch
  std::memcpy(v_feats[wb].raw_mutable_tensor(off), data, sz);

  // free Cassandra result memory (data included)
  cass_result_free(result);
}

void BatchLoader::copy_data_int(const CassResult* result,
                              const cass_byte_t* data, size_t sz,
                              cass_int32_t lab, int off, int wb) {
  // wait for feature tensor to be allocated
  {
    std::unique_lock<std::mutex> lck(alloc_mtx[wb]);
    while (copy_jobs[wb].size() != bs[wb]) {
      alloc_cv[wb].wait(lck);
    }
  }
  // copy data in batch
  std::memcpy(v_feats[wb].raw_mutable_tensor(off), data, sz);
  std::memcpy(v_labs[wb].raw_mutable_tensor(off), &lab, sizeof(INT_LABEL_T));

  // free Cassandra result memory (data included)
  cass_result_free(result);
}

void BatchLoader::copy_data_img(const CassResult* result,
                                const cass_byte_t* data, size_t sz,
                                const cass_byte_t* lab, size_t l_sz,
                                int off, int wb) {
  // wait for feature tensor to be allocated
  {
    std::unique_lock<std::mutex> lck(alloc_mtx[wb]);
    while (copy_jobs[wb].size() != bs[wb]) {
      alloc_cv[wb].wait(lck);
    }
  }
  // copy data in batch
  std::memcpy(v_feats[wb].raw_mutable_tensor(off), data, sz);
  std::memcpy(v_labs[wb].raw_mutable_tensor(off), lab, l_sz);

  // free Cassandra result memory (data included)
  cass_result_free(result);
}

void BatchLoader::transfer2copy(CassFuture* query_future, int wb, int i) {
  const CassResult* result = cass_future_get_result(query_future);
  if (result == NULL) {
    // Handle error
    const char* error_message;
    size_t error_message_length;
    cass_future_error_message(query_future,
                              &error_message, &error_message_length);
    fprintf(stderr, "Unable to run query: '%.*s'\n",
            static_cast<int>(error_message_length), error_message);
    cass_future_free(query_future);
    throw std::runtime_error("Error: unable to execute query");
  }
  // decode result
  const CassRow* row = cass_result_first_row(result);
  if (row == NULL) {
    // Handle error
    throw std::runtime_error("Error: query returned empty set");
  }
  // feature
  const CassValue* c_data =
    cass_row_get_column_by_name(row, data_col.c_str());
  const cass_byte_t* data;
  size_t sz;
  cass_value_get_bytes(c_data, &data, &sz);
  shapes[wb][i] = sz;
  // label/mask/none
  std::future<void> cj;
  if (label_t == lab_none) {
    // enqueue image copy
    cj = copy_pool->enqueue(&BatchLoader::copy_data_none, this,
                            result, data, sz, i, wb);
  } else if (label_t == lab_int) {
    const CassValue* c_lab =
      cass_row_get_column_by_name(row, label_col.c_str());
    cass_int32_t lab;
    cass_value_get_int32(c_lab, &lab);
    // enqueue image copy + int label
    cj = copy_pool->enqueue(&BatchLoader::copy_data_int, this,
                            result, data, sz, lab, i, wb);
  } else if (label_t == lab_img) {
    const CassValue* c_lab =
      cass_row_get_column_by_name(row, label_col.c_str());
    const cass_byte_t* lab;
    size_t l_sz;
    cass_value_get_bytes(c_lab, &lab, &l_sz);
    lab_shapes[wb][i] = l_sz;
    // enqueue image copy + image label (e.g., mask)
    cj = copy_pool->enqueue(&BatchLoader::copy_data_img, this,
                            result, data, sz, lab, l_sz, i, wb);
  }
  // saving raw image size
  {
    std::unique_lock<std::mutex> lck(alloc_mtx[wb]);
    copy_jobs[wb].emplace_back(std::move(cj));
    // if all copy_jobs added
    if (copy_jobs[wb].size() == bs[wb]) {
      // allocate feature tensor and notify waiting threads
      dali::TensorListShape t_sz(shapes[wb], bs[wb], 1);
      v_feats[wb].Resize(t_sz, DALI_IMG_TYPE);
      if (label_t == lab_img) {
        dali::TensorListShape t_sz(lab_shapes[wb], bs[wb], 1);
        v_labs[wb].Resize(t_sz, DALI_IMG_TYPE);
      }
      alloc_cv[wb].notify_all();
    }
  }
}

void BatchLoader::wrap_enq(CassFuture* query_future, void* v_fd) {
  futdata* fd = static_cast<futdata*>(v_fd);
  BatchLoader* batch_ldr = fd->batch_ldr;
  int wb = fd->wb;
  int i = fd->i;
  delete(fd);
  if (batch_ldr->ooo) {
    batch_ldr->ooo_enqueue(query_future);
  } else {
    batch_ldr->transfer2copy(query_future, wb, i);
  }
}

void BatchLoader::ooo_enqueue(CassFuture* query_future) {
  // insert future and check if there are now enough to create a new batch
  ooo_buf_mtx.lock();
  int wb = ooo_buf.front();
  int bsz = bs[wb];
  int i = ooo_in_bs[wb];
  transfer2copy(query_future, wb, i);
  ooo_in_bs[wb] = ++i;
  if (i == bsz) {
    ooo_in_bs[wb] = 0;
    ooo_buf.pop();
  }
  ooo_buf_mtx.unlock();
}

void BatchLoader::keys2transfers(const std::vector<CassUuid>& keys, int wb) {
  // start all transfers in parallel (send requests to driver)
  for (size_t i=0; i != keys.size(); ++i) {
    CassUuid cuid = keys[i];
    // prepare query
    CassStatement* statement = cass_prepared_bind(prepared);
    cass_statement_bind_uuid_by_name(statement, id_col.c_str(), cuid);
    CassFuture* query_future = cass_session_execute(session, statement);
    cass_statement_free(statement);
    futdata* fd = new futdata();
    fd->batch_ldr = this;
    fd->wb = wb;
    fd->i = i;
    cass_future_set_callback(query_future, wrap_enq, fd);
    cass_future_free(query_future);
  }
}

std::future<BatchImgLab> BatchLoader::start_transfers(
                             const std::vector<CassUuid>& keys, int wb) {
  bs[wb] = keys.size();
  copy_jobs[wb].reserve(bs[wb]);
  allocTens(wb);  // allocate space for tensors
  if (ooo) {  // out-of-order?
    std::unique_lock<std::mutex> lck(ooo_buf_mtx);
    ooo_buf.push(wb);
  }
  // enqueue keys for transfers
  comm_job[wb] = comm_pool->enqueue(
                             &BatchLoader::keys2transfers, this, keys, wb);
  auto r = wait_pool->enqueue(&BatchLoader::wait4images, this, wb);
  return(r);
}

BatchImgLab BatchLoader::wait4images(int wb) {
  // check if tranfers started
  comm_job[wb].get();
  // wait for all copy_jobs to be scheduled
  {
    std::unique_lock<std::mutex> lck(alloc_mtx[wb]);
    while (copy_jobs[wb].size() != bs[wb]) {
      alloc_cv[wb].wait(lck);
    }
  }
  // check if all images were copied correctly
  for (auto it=copy_jobs[wb].begin(); it != copy_jobs[wb].end(); ++it) {
    it->get();  // using get instead of wait, to propagates exceptions
  }
  // reset job queues
  copy_jobs[wb].clear();
  comm_job[wb] = std::future<void>();
  // copy vector to be returned
  BatchRawImage nv_feats = std::move(v_feats[wb]);
  BatchLabel nv_labs = std::move(v_labs[wb]);
  BatchImgLab r = std::make_pair(std::move(nv_feats), std::move(nv_labs));
  return(r);
}

void BatchLoader::check_connection() {
  if (!connected) {
    connect();
    connected = true;
  }
}

void BatchLoader::prefetch_batch(const std::vector<CassUuid>& ks) {
  int wb = write_buf.front();
  write_buf.pop();
  check_connection();
  batch[wb] = start_transfers(ks, wb);
  read_buf.push(wb);
}

BatchImgLab BatchLoader::blocking_get_batch() {
  // recover
  int rb = read_buf.front();
  read_buf.pop();
  auto r = batch[rb].get();
  write_buf.push(rb);
  return(r);
}

void BatchLoader::ignore_batch() {
  if (!connected) {
    return;
  }
  // wait for flying batches to be retrieved
  while (!read_buf.empty()) {
    auto b = blocking_get_batch();
  }
  return;
}

}  // namespace crs4
