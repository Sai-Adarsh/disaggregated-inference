# Disaggregated prefill and decode

The synthetic decoder has two attention layers, four heads per layer, a hidden width of 64, and a 27-byte output vocabulary. Prefill applies dense Q/K/V projections, causal scaled dot-product softmax, and a residual output projection at every prompt position. The second layer consumes the first layer's output. The split worker receives both layers' K/V and the last hidden row. The paired command checks greedy output against a local run for the chosen prompt and backend. The recorded CPU loopback matched exactly. Weights are deterministic and untrained.

The KV03 request has three network-order 32-bit fields for version, prompt length, and generation length, followed by network-order FP32 arrays. For a prompt with `T` bytes the state payload is `4 × (2 × 2 × T × 64 + 64)` bytes, plus the 12-byte header. The worker streams one byte per token then sends two network-order 32-bit fields, the `DONE` marker and the emitted token count, after it finishes decoding. Upgrade both workers together. A client with missing or invalid confirmation exits with an error, even if it has already printed tokens.

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

`compare` drops one warmup pair and alternates run order. It exits on the first mismatch and writes two CSV rows per confirmed pair. Both `ttft_ms` and `total_ms` start at client prefill. For the split path, TTFT stops at the first token and total time stops at a valid completion record. `connect_send_ms` includes connection setup, serialization, and the client send. The worker times receive and decode locally. The monolithic GPU path reuses device-resident K/V. The split GPU path stages the state in host memory.

`serve` accepts one request. `serve-loop` takes 1–64 worker threads and leaves excess connections in the transport backlog. Established TCP connections have a 30-second idle read and write timeout on both endpoints. Multiple CUDA handlers share the default device stream.

### Recorded loopback run

Apple M3 Pro, macOS 27.2.1, Apple Clang 21.0.0, Release build, both processes on the same host. All 48 emitted tokens agreed in each of nine measured KV03 pairs. Each split request transferred 295,168 bytes of model state plus the 12-byte header. Each response returned 48 token bytes and 8 completion bytes. [Raw samples](measurements/m3-pro-tcp-loopback.csv)

| Path | Median prefill | Median TTFT | Median total |
|---|---:|---:|---:|
| Monolithic | 6.774 ms | 6.786 ms | 8.496 ms |
| Split over TCP | 6.878 ms | 7.742 ms | 10.345 ms |

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

## Tests and coverage

The suite starts TCP workers on ephemeral loopback ports. It checks causal prefix invariance, streamed tokens, split-versus-local parity, invalid frames, idle timeouts on both endpoints, and signal interruption. It sends complete token streams with missing, truncated, and invalid completion frames and checks that the client rejects them. For a bounded worker run, pass a connection limit after the backend. Server port `0` asks the OS for a free port, which appears in the worker log. The worker drains accepted connections before exiting.

```sh
cmake -S . -B build/coverage -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping"
cmake --build build/coverage -j
profiles="$(mktemp -d)"
LLVM_PROFILE_FILE="$profiles/%m-%p.profraw" ctest --test-dir build/coverage --output-on-failure
xcrun llvm-profdata merge -sparse "$profiles"/*.profraw -o "$profiles/merged.profdata"
xcrun llvm-cov report build/coverage/split_inference \
  -object build/coverage/split_inference_tests -instr-profile="$profiles/merged.profdata" \
  --show-branch-summary main.cpp model.cpp transport.cpp model.hpp feature.hpp transport.hpp
```

On Linux, use `llvm-profdata` and `llvm-cov` from the same Clang toolchain without the `xcrun` prefix. The AppleClang CPU run covered 561/566 lines (99.12%) and 182/214 branches (85.05%). Uncovered CPU lines include transport initialization failure after accept and a zero-byte send. CUDA and RDMA verification require separate builds and hardware.
