# Disaggregated prefill and decode

The synthetic decoder has two attention layers, four heads per layer, a hidden width of 64, and a 27-byte output vocabulary. Prefill applies dense Q/K/V projections, causal scaled dot-product softmax, and a residual output projection at every prompt position. The second layer consumes the first layer's output. The split worker receives both layers' K/V and the last hidden row, then emits the same greedy tokens as a local run. Weights are deterministic and untrained.

The KV02 frame contains the version, prompt length and generation length, followed by network-order FP32 arrays. Both sides must use KV02. For a prompt with `T` bytes the state payload is `4 × (2 × 2 × T × 64 + 64)` bytes, plus a 12-byte header. Prefill performs quadratic attention work. Each new generated token attends over the cached prefix once per layer.

## Compare local and split execution

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/split_inference serve-loop tcp 127.0.0.1 8123 2
```

In a second terminal:

```sh
./build/split_inference compare tcp 127.0.0.1 8123 "hello inference" 24 9 > comparison.csv
./build/split_inference prefill tcp 127.0.0.1 8123 "hello inference" 24
```

`compare` drops one warmup pair and alternates run order. It exits on the first token mismatch and writes two CSV rows per completed pair. `ttft_ms` and `total_ms` run from client-side prefill start. `connect_send_ms` includes TCP/RDMA connection setup, serialization and the client send. It is not a measure of one-way network latency. The worker logs its own receive and decode intervals. The monolithic GPU path reuses device-resident K/V; the split GPU path stages the state on host memory.

`serve` accepts one request. `serve-loop` takes 1–64 worker threads and leaves excess connections in the transport backlog. Persistent TCP connections have a 30-second idle I/O timeout. Multiple CUDA handlers share the default device stream.

### Recorded loopback run

Apple M3 Pro, macOS 27.2.1, Apple Clang 21.0.0, Release build, both processes on the same host. All 48 emitted tokens agreed in each of nine measured pairs. Each split request transferred 295,168 bytes of model state plus the 12-byte header. See [the raw samples](measurements/m3-pro-tcp-loopback.csv).

| Path | Median prefill | Median TTFT | Median total |
|---|---:|---:|---:|
| Monolithic | 6.780 ms | 6.788 ms | 8.532 ms |
| Split over TCP | 6.770 ms | 7.514 ms | 10.399 ms |

To reproduce the 288-byte prompt, use this client command with the same worker above:

```sh
unit=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
./build/split_inference compare tcp 127.0.0.1 8123 "$unit$unit$unit$unit" 48 9 cpu
```

## H100 to L40 experiment

Use Linux, CUDA, rdma-core development files, and a working InfiniBand or RoCE route. Replace the example decode host address. Use a trusted lab network.

```sh
cmake -S . -B build-rdma -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
  -DENABLE_RDMA=ON '-DCMAKE_CUDA_ARCHITECTURES=89;90'
cmake --build build-rdma -j

# L40
./build-rdma/split_inference serve-loop rdma 10.20.0.2 8123 1 cuda

# H100
./build-rdma/split_inference compare rdma 10.20.0.2 8123 "hello inference" 24 9 cuda > comparison.csv
```

The rsocket path transfers host buffers, so this experiment includes GPU-to-host and host-to-GPU copies. The H100/L40 comparison above has not been run. It measures the configured two-host split against the H100-only baseline; repeat with a colocated TCP control and record hardware, link, and prompt length before attributing differences to RDMA. This fixture does not measure checkpoint throughput, saturated model FLOP utilization, GPUDirect RDMA, or continuous batching.

[rdma-core rsocket API](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/man/rsocket.7.in)
