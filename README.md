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

## Serve repeated requests

Start a persistent worker with two request handlers.

```sh
./build/split_inference serve-loop tcp 127.0.0.1 8123 2
```

From another terminal, send two concurrent requests and then a third request to the same worker.

```sh
./build/split_inference prefill tcp 127.0.0.1 8123 "first request" 12 > first.txt &
./build/split_inference prefill tcp 127.0.0.1 8123 "second request" 12 > second.txt &
wait
cat first.txt second.txt
./build/split_inference prefill tcp 127.0.0.1 8123 "third request" 12
```

Set the worker count between 1 and 64. At capacity, the service leaves additional connections in the transport backlog. A failed request releases its handler, and each successful request writes its decode timings to the worker log. Persistent TCP connections have a 30-second idle I/O timeout while a request is in progress. Stop the local worker with Ctrl-C.

The persistent mode accepts the same RDMA and CUDA choices as one-shot mode, for example `serve-loop rdma 10.20.0.2 8123 2 cuda`. The worker count controls request concurrency across the connections. CUDA requests share the existing default device stream.

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

The transport uses librdmacm rsockets and stages KV tensors in host memory on both sides. A trained checkpoint, continuous GPU batching, GPUDirect RDMA, and saturated model FLOP utilization measurements require further integration.

[rdma-core rsocket protocol](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/docs/rsocket) · [rdma-core rsocket API](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/man/rsocket.7.in)
