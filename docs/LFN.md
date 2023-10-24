# Long fat networks

Let's explore how to optimize our data loader for use across long fat
networks, i.e., networks that have a high [bandwidth-delay
product](https://en.wikipedia.org/wiki/Bandwidth-delay_product), e.g.,
100 ms latency and 10 Gb/s bandwidth.

For instance, imagine a setup where you have your Cassandra DB,
containing the required training images in datacenter A, while the
computing nodes with the GPUs are located in datacenter B, which may
even be far away in a different country.

To take advantage of such networks, it is crucial to have a deep
prefetch queue that can be processed in parallel. To this purporse,
our plugin provides the following configurable parameters:

- `prefetch_buffers`: the plugin employs multi-buffering, to hide the
  network latencies. Default: 2.
- `io_threads`: number of IO threads used by the Cassandra driver
  (which also limits the number of TCP connections). Default: 2.
- `comm_threads`: number of threads handling the
  communications. Default: 2.
- `copy_threads`: number of threads copying the data. Default: 2.

As an example, we loaded the original ImageNet dataset over a 25 GbE
network with an artificial latency of 100ms (set with `tc-netem`, with
no packet loss), using a `batch_size` of 512 and without any decoding
or preprocessing. Our test nodes (equipped with an Intel Xeon CPU
E5-2650 v4 @ 2.20GHz), achieved about 40 batches per second, which
translates to more than 20,000 images per second and a throughput of
roughly 20 Gb/s. Note that this throughput refers to a single python process,
and that in [a distributed training](examples/imagenette/README.md#multi-gpu-training)
there is such a process *for each GPU*. We used the following
parameters for the test:

- `prefetch_buffers`: 16
- `io_threads`: 8
- `comm_threads`: 1
- `copy_threads`: 4

## Handling variance and packet loss

When sending packets at large distance across the internet it is
common to experience packet loss due to congested routes. This can
significantly impact throughput, especially when requesting a sequence
of transfers, as a delay in one transfer can stall the entire
pipeline. Prefetching can exacerbate this issue by producing an
initial burst of requests, leading to even higher packet loss.

To address these problems and enable high-bandwidth transfers over
long distances (i.e., high latencies), we have extended our code in
two ways:

1. We have developed an out-of-order version of the data loader that
   can be activated by setting `ooo=True`. This version of the loader
   returns the images as soon as they are received, *potentially
   altering their sequence and mixing different batches*.
2. We have implemented a parametrized diluted prefetching method that
   requests an additional image every `n` normal requests, thus
   limiting the initial burst. To activate it, set `slow_start=4`, for
   example.
