// Copyright 2021-2 CRS4
// 
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "batchhandler.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <numeric>


BatchHandler::~BatchHandler(){
  if (connected){
    cass_session_free(session);
    cass_cluster_free(cluster);
    delete(conv_pool);
    delete(comm_pool);
    delete(batch_pool);
  }
}

void BatchHandler::connect(){
  cass_cluster_set_contact_points(cluster, s_cassandra_ips.c_str());
  cass_cluster_set_credentials(cluster, username.c_str(), password.c_str());
  cass_cluster_set_port(cluster, port);
  cass_cluster_set_protocol_version(cluster, CASS_PROTOCOL_VERSION_V4);
  cass_cluster_set_num_threads_io(cluster, tcp_connections);
  cass_cluster_set_queue_size_io(cluster, 65536); // max pending requests
  CassFuture* connect_future = cass_session_connect(session, cluster);
  CassError rc = cass_future_error_code(connect_future);
  cass_future_free(connect_future);
  if (rc != CASS_OK) {
    throw std::runtime_error("Error: unable to connect to Cassandra DB. ");
  }
  // assemble query
  std::stringstream ss;
  ss << "SELECT " << label_col << ", " << data_col <<
    " FROM " << table << " WHERE " << id_col << "=?" << std::endl;
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
  comm_pool = new ThreadPool(comm_par);
  conv_pool = new ThreadPool(threads);
  batch_pool = new ThreadPool(batch_par);
}

BatchHandler::BatchHandler(std::string table, std::string label_col,
			   std::string data_col, std::string id_col,
			   // std::vector<int> label_map,
			   std::string username, std::string cass_pass,
			   std::vector<std::string> cassandra_ips,
			   int tcp_connections, int threads, int batch_par,
			   int comm_par, int prefetch_buffers, int port) :
  table(table), label_col(label_col), data_col(data_col), id_col(id_col),
  // label_map(label_map),
  username(username), password(cass_pass),
  cassandra_ips(cassandra_ips), port(port),
  tcp_connections(tcp_connections), threads(threads),
  batch_par(batch_par), comm_par(comm_par), prefetch_buffers(prefetch_buffers)
{
  // init multi-buffering variables
  bs.resize(prefetch_buffers);
  batch.resize(prefetch_buffers);
  conv_jobs.resize(prefetch_buffers);
  comm_jobs.resize(prefetch_buffers);
  v_feats.resize(prefetch_buffers);
  v_labs.resize(prefetch_buffers);
  alloc_mtx = std::vector<std::mutex>(prefetch_buffers);
  loc_mtx = std::vector<std::mutex>(prefetch_buffers);
  wait_mtx = std::vector<std::mutex>(prefetch_buffers);
  for(int i=0; i<prefetch_buffers; ++i){
    write_buf.push(i);
    allocated.push_back(false);
  }
  // join cassandra ip's into comma seperated string
  s_cassandra_ips =
    std::accumulate(cassandra_ips.begin(), cassandra_ips.end(),
	    std::string(), 
	    [](const std::string& a, const std::string& b) -> std::string { 
		      return a + (a.length() > 0 ? "," : "") + b; 
	    });
  // transforming labels?
  // use_label_map = !label_map.empty();
}

void BatchHandler::allocTens(int wb){
  v_feats[wb].resize(bs[wb]);

  v_labs[wb] = BatchLabel();
  std::vector<long int> v_sz(bs[wb], 1);
  ::dali::TensorListShape t_sz(v_sz, bs[wb], 1);
  v_labs[wb].Resize(t_sz, ::dali::DALI_INT32);
}

void BatchHandler::img2tensor(const CassResult* result,
			      const cass_byte_t* data, size_t sz,
			      cass_int32_t lab, int off, int wb){
  // copy buffer as RawImage
  RawImage buf(data, data+sz);
  // free Cassandra result memory (data included)
  cass_result_free(result);  
  // do something with buf and lab
  ////////////////////////////////////////////////////////////////////////
  // run once per batch, allocate new vector
  ////////////////////////////////////////////////////////////////////////
  alloc_mtx[wb].lock();
  // allocate batch
  if (!allocated[wb]){
    allocTens(wb);
    allocated[wb] = true;
  }
  alloc_mtx[wb].unlock();
  // insert buffer in batch
  v_feats[wb][off] = buf;
  *(v_labs[wb].mutable_tensor<Label>(off)) = lab;
}

void BatchHandler::transfer2conv(CassFuture* query_future, int wb, int i){
  const CassResult* result = cass_future_get_result(query_future);
  if (result == NULL) {
    // Handle error
    const char* error_message;
    size_t error_message_length;
    cass_future_error_message(query_future,
			      &error_message, &error_message_length);
    cass_future_free(query_future);
    throw std::runtime_error("Error: unable to execute query, " +
			std::string(error_message));
  }
  // decode result
  const CassRow* row = cass_result_first_row(result);
  if (row == NULL) {
    // Handle error
    throw std::runtime_error("Error: query returned empty set");
  }
  const CassValue* c_lab =
    cass_row_get_column_by_name(row, label_col.c_str());
  const CassValue* c_data =
    cass_row_get_column_by_name(row, data_col.c_str());
  cass_int32_t lab;
  cass_value_get_int32(c_lab, &lab);
  const cass_byte_t* data;
  size_t sz;
  cass_value_get_bytes(c_data, &data, &sz);
  // enqueue image conversion
  auto cj = conv_pool->enqueue(&BatchHandler::img2tensor, this,
			       result, data, sz, lab, i, wb);
  loc_mtx[wb].lock();
  conv_jobs[wb].emplace_back(std::move(cj));
  // if all conv_jobs added, unlock wait mutex
  if (conv_jobs[wb].size()==bs[wb])
    wait_mtx[wb].unlock();    
  loc_mtx[wb].unlock();
}

void BatchHandler::wrap_t2c(CassFuture* query_future, void* v_fd){
  futdata* fd = static_cast<futdata*>(v_fd);
  BatchHandler* bh = fd->bh;
  int wb = fd->wb; 
  int i = fd->i;
  delete(fd);
  bh->transfer2conv(query_future, wb, i);
}

void BatchHandler::keys2transfers(const std::vector<std::string>& keys, int wb){
  // start all transfers in parallel
  for(size_t i=0; i!=keys.size(); ++i){
    std::string id = keys[i];
    // prepare query
    CassStatement* statement = cass_prepared_bind(prepared);
    CassUuid cuid;
    cass_uuid_from_string(id.c_str(), &cuid);
    cass_statement_bind_uuid_by_name(statement, id_col.c_str(), cuid);
    CassFuture* query_future = cass_session_execute(session, statement);
    cass_statement_free(statement);
    futdata* fd = new futdata();
    fd->bh=this; fd->wb=wb; fd->i=i;
    cass_future_set_callback(query_future, wrap_t2c, fd);
    cass_future_free(query_future);
  }
}

std::future<BatchImgLab> BatchHandler::start_transfers(const std::vector<std::string>& keys, int wb){
  // lock until conv_jobs have been added
  wait_mtx[wb].lock();
  bs[wb] = keys.size();
  conv_jobs[wb].reserve(bs[wb]);
  // enqueue keys for transfers
  comm_jobs[wb] = comm_pool->enqueue(&BatchHandler::keys2transfers, this, keys, wb);
  auto r = batch_pool->enqueue(&BatchHandler::wait4images, this, wb);
  return(r);
}

BatchImgLab BatchHandler::wait4images(int wb){
  // check in tranfers succeeded
  comm_jobs[wb].get();
  // wait for conv_jobs to be scheduled
  wait_mtx[wb].lock();
  // check if all images were converted correctly
  for(auto it=conv_jobs[wb].begin(); it!=conv_jobs[wb].end(); ++it){
    it->get(); // using get to propagates exceptions
  }
  // reset job queues
  conv_jobs[wb].clear();
  comm_jobs[wb] = std::future<void>();
  // copy vector to be returned
  BatchRawImage nv_feats = std::move(v_feats[wb]);
  BatchLabel nv_labs = std::move(v_labs[wb]);
  allocated[wb] = false;
  wait_mtx[wb].unlock(); // release lock on batch wb
  BatchImgLab r = std::make_pair(std::move(nv_feats), std::move(nv_labs));
  return(r);
}

void BatchHandler::check_connection(){
  if(!connected){
    connect();
    connected = true;
  }
}

void BatchHandler::prefetch_batch(const std::vector<std::string>& ks){
  int wb = write_buf.front();
  write_buf.pop();
  check_connection();
  batch[wb] = start_transfers(ks, wb);  
  read_buf.push(wb);
}

BatchImgLab BatchHandler::blocking_get_batch(){
  // recover
  int rb = read_buf.front();
  read_buf.pop();
  auto r = batch[rb].get();
  write_buf.push(rb);
  return(r);
}

void BatchHandler::ignore_batch(){
  if (!connected){
    return;
  }
  // wait for flying batches to be computed
  while(!read_buf.empty()){
    auto b = blocking_get_batch();
  }
  return;
}
