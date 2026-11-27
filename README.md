# Disaggregated prefill and decode

Two C++ processes split causal attention across prefill and decode. Prefill computes key and value tensors and sends them to the worker. The worker attends over the transferred cache, appends generated state, and streams tokens back. Each process can use CPU or CUDA independently. The model uses deterministic synthetic weights.

## Run over TCP

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the worker in one terminal and prefill in another.

```sh
./build/split_inference serve tcp 127.0.0.1 8123
./build/split_inference prefill tcp 127.0.0.1 8123 "hello inference" 24
```

The worker handles one request. Prefill prints the fixture tokens and reports the KV payload size, total latency, and time from prefill start to first token. Both processes write timing metrics to standard error.

## Run across an H100 and L40

Use Linux, the CUDA toolkit, rdma-core development files, and an InfiniBand or RoCE route between the hosts. Replace `10.20.0.2` with the decode worker's IPv4 address on that route. Use a trusted lab network.

```sh
# Build on both hosts.
cmake -S . -B build-rdma -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
  -DENABLE_RDMA=ON '-DCMAKE_CUDA_ARCHITECTURES=89;90'
cmake --build build-rdma -j

# Decode worker on the L40.
./build-rdma/split_inference serve rdma 10.20.0.2 8123 cuda

# Prefill client on the H100.
./build-rdma/split_inference prefill rdma 10.20.0.2 8123 "hello inference" 24 cuda
```

The transport uses librdmacm rsockets and stages KV tensors in host memory on both sides. A trained checkpoint, concurrent scheduling, GPUDirect RDMA, and saturated model FLOP utilization measurements require further integration.

[rdma-core rsocket protocol](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/docs/rsocket) · [rdma-core rsocket API](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/man/rsocket.7.in)
