# Disaggregated prefill and decode reference

Run the commands from the repository root.

Run prefill on one process and decode on another. The paired benchmark compares the split path with monolithic execution using the same prompt and backend.

## Run the TCP fixture

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

`compare` drops one warmup pair, alternates run order, and writes two CSV rows per confirmed pair. It exits on the first mismatch.

### Timing columns

- `ttft_ms` and `total_ms` start at client prefill. In a split run, TTFT ends at the first token and total time ends at a valid completion record.
- `connect_send_ms` covers connection setup, serialization, and the client send. The worker times receive and decode locally.
- Monolithic GPU execution reuses device-resident K/V. Split GPU execution stages the state in host memory.

### Worker limits

`serve` accepts one request. `serve-loop` takes 1–64 worker threads and leaves excess connections in the transport backlog. TCP clients have a 30-second connection deadline. Both endpoints use a 30-second read and write idle timeout once connected. CUDA handlers share the default device stream.

Pass a connection limit after the backend to bound a worker run. Port `0` requests a free port from the OS, which the worker logs. The worker drains accepted connections before exiting.

### Recorded loopback run

Both processes ran on an Apple M3 Pro with macOS 27.2.1 and Apple Clang 21.0.0 in a Release build. All 48 tokens matched in each of nine measured KV03 pairs. Each split request sent 295,168 bytes of model state plus the 12-byte header. Each response sent 48 token bytes and 8 completion bytes. [Raw samples](../measurements/m3-pro-tcp-loopback.csv)

| Path | Median prefill | Median TTFT | Median total |
|---|---:|---:|---:|
| Monolithic | 6.774 ms | 6.786 ms | 8.496 ms |
| Split over TCP | 6.878 ms | 7.742 ms | 10.345 ms |

Use the same worker to reproduce the 288-byte prompt.

```sh
unit=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
./build/split_inference compare tcp 127.0.0.1 8123 "$unit$unit$unit$unit" 48 9 cpu
```

## Model and KV03 wire format

The synthetic decoder uses deterministic, untrained weights. It has two attention layers, four heads per layer, a hidden width of 64, and a 27-byte output vocabulary. At each prompt position, prefill applies dense Q/K/V projections, causal scaled dot-product softmax, and a residual output projection. The second layer consumes the first layer's output.

- **Request header** uses three network-order 32-bit fields for version, prompt length, and generation length.
- **Model state** follows as network-order FP32 arrays containing both layers' K/V and the last hidden row. For a prompt of `T` bytes, the payload is `4 × (2 × 2 × T × 64 + 64)` bytes, plus the 12-byte header.
- **Response** streams one byte per token. After decode, the worker sends the `DONE` marker and the emitted token count as two network-order 32-bit fields.

Upgrade both workers together. Missing or invalid confirmation makes the client exit with an error, even after it has printed tokens.

## Run the H100-to-L40 RDMA experiment

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

The H100/L40 comparison has no recorded results. The procedure compares the two-host split with an H100-only baseline. The rsocket path transfers host buffers, so timings include GPU-to-host and host-to-GPU copies. Repeat with a colocated TCP control. Record hardware, link, and prompt length before attributing differences to RDMA.

Checkpoint throughput, saturated model FLOP utilization, GPUDirect RDMA, and continuous batching are outside the fixture's scope.

[rdma-core rsocket API](https://github.com/linux-rdma/rdma-core/blob/master/librdmacm/man/rsocket.7.in)

## Tests and coverage

The suite starts TCP workers on ephemeral loopback ports.

- Causal prefix invariance, streamed tokens, split-versus-local parity, and invalid frames.
- Idle timeouts at both endpoints, interrupted syscalls, and broken-pipe errors after a peer reset.
- Client rejection of missing, truncated, or invalid completion frames after a complete token stream.
- Client timeout when the Linux accept queue is full.

### Coverage command

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

On Linux, use `llvm-profdata` and `llvm-cov` from the same Clang toolchain without the `xcrun` prefix.

The recorded AppleClang CPU run covered 561/566 lines (99.12%) and 182/214 branches (85.05%). Uncovered CPU lines include transport initialization failure after accept and a zero-byte send. CUDA and RDMA verification require separate builds and hardware.
