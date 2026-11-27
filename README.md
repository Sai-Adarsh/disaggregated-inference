# Disaggregated prefill and decode

A CPU demo runs prefill and decode in separate processes and streams generated tokens over TCP.

From the repo root, build and start the worker with CMake 3.21+ and a C++20 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/split_inference serve-loop tcp 127.0.0.1 8123 2
```

From the repo root in a second terminal, send a prompt.

```sh
./build/split_inference prefill tcp 127.0.0.1 8123 "hello inference" 24
```

[Benchmarks, RDMA, and tests](docs/reference.md)
