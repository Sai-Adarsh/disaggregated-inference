#include "model.hpp"
#include "transport.hpp"

#include <arpa/inet.h>

#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t protocol = 0x4b563031;  // KV01

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

int parse_int(std::string_view text, int min, int max) {
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < min || value > max) {
        throw std::invalid_argument("numeric argument outside expected range");
    }
    return value;
}

bool use_cuda(std::string_view backend) {
    if (backend == "cpu") {
        return false;
    }
    if (backend == "cuda") {
#ifdef INFERENCE_CUDA
        return true;
#else
        throw std::invalid_argument("rebuild with ENABLE_CUDA=ON to use CUDA");
#endif
    }
    throw std::invalid_argument("expected cpu or cuda backend");
}

void write_cache(const Stream& stream, const fixture::Cache& cache, int steps) {
    const std::array header{htonl(protocol), htonl(static_cast<std::uint32_t>(cache.size())),
                            htonl(steps), htonl(cache.last)};
    stream.send(header.data(), sizeof(header));
    std::vector<std::uint32_t> payload;
    payload.reserve(cache.keys.size() + cache.values.size());
    for (const auto& matrix : {&cache.keys, &cache.values}) {
        for (const float value : *matrix) {
            payload.push_back(htonl(std::bit_cast<std::uint32_t>(value)));
        }
    }
    stream.send(payload.data(), payload.size() * sizeof(payload[0]));
}

fixture::Cache read_cache(const Stream& stream, int& steps) {
    std::array<std::uint32_t, 4> header{};
    stream.receive(header.data(), sizeof(header));
    const auto rows = ntohl(header[1]);
    const auto requested_steps = ntohl(header[2]);
    const auto last = ntohl(header[3]);
    if (ntohl(header[0]) != protocol || rows == 0 || rows > fixture::max_prompt
        || requested_steps == 0 || requested_steps > fixture::max_generated || last > 255) {
        throw std::runtime_error("incompatible KV handoff header");
    }
    steps = static_cast<int>(requested_steps);
    const auto matrix_elements = std::size_t{rows} * fixture::width;
    std::vector<std::uint32_t> payload(2 * matrix_elements);
    stream.receive(payload.data(), payload.size() * sizeof(payload[0]));
    fixture::Cache cache;
    cache.keys.resize(matrix_elements);
    cache.values.resize(matrix_elements);
    cache.last = static_cast<unsigned char>(last);
    for (std::size_t i = 0; i < matrix_elements; ++i) {
        const auto key = std::bit_cast<float>(ntohl(payload[i]));
        const auto value = std::bit_cast<float>(ntohl(payload[matrix_elements + i]));
        if (!std::isfinite(key) || !std::isfinite(value) || std::abs(key) > 1 || std::abs(value) > 1) {
            throw std::runtime_error("invalid fixture KV entry");
        }
        cache.keys[i] = key;
        cache.values[i] = value;
    }
    return cache;
}

void serve(Stream::Kind transport, std::string_view address, int port, bool gpu) {
    const auto listener = Stream::listen(transport, address, port);
    std::cerr << "listening on " << address << ':' << port << '\n';
    const auto client = listener.accept();
    const auto start = Clock::now();
    int steps = 0;
    auto cache = read_cache(client, steps);
    const auto received = Clock::now();
    const auto emit = [&client](unsigned char token) { client.send(&token, 1); };
#ifdef INFERENCE_CUDA
    if (gpu) {
        fixture::decode_cuda(cache, steps, emit);
    } else
#endif
    {
        fixture::decode_cpu(std::move(cache), steps, emit);
    }
    const auto finished = Clock::now();
    std::cerr << "backend=" << (gpu ? "cuda" : "cpu")
              << " decode_tokens=" << steps << " receive_ms=" << milliseconds(received - start)
              << " decode_ms=" << milliseconds(finished - received) << '\n';
}

void prefill(Stream::Kind transport, std::string_view address, int port,
             std::string_view prompt, int steps, bool gpu) {
    if (prompt.empty() || prompt.size() > fixture::max_prompt) {
        throw std::invalid_argument("prompt must contain 1–8192 bytes");
    }

    const auto start = Clock::now();
    const auto cache =
#ifdef INFERENCE_CUDA
        gpu ? fixture::prefill_cuda(prompt) :
#endif
        fixture::prefill_cpu(prompt);
    const auto computed = Clock::now();
    const auto stream = Stream::connect(transport, address, port);
    write_cache(stream, cache, steps);

    unsigned char next = 0;
    stream.receive(&next, 1);
    const auto first_token = Clock::now();
    std::cout << next << std::flush;
    for (int i = 1; i < steps; ++i) {
        stream.receive(&next, 1);
        std::cout << next << std::flush;
    }
    const auto finished = Clock::now();
    std::cout << '\n';
    std::cerr << "backend=" << (gpu ? "cuda" : "cpu")
              << " prompt_bytes=" << prompt.size() << " kv_bytes=" << 2 * cache.keys.size() * sizeof(float)
              << " prefill_ms=" << milliseconds(computed - start)
              << " ttft_ms=" << milliseconds(first_token - start)
              << " total_ms=" << milliseconds(finished - start) << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    std::signal(SIGPIPE, SIG_IGN);
    if (argc < 5) {
        throw std::invalid_argument("usage: split_inference serve tcp|rdma IP PORT [cpu|cuda]\n"
                                    "       split_inference prefill tcp|rdma IP PORT PROMPT STEPS [cpu|cuda]");
    }
    const std::string_view mode = argv[1];
    const std::string_view protocol_name = argv[2];
    if (protocol_name != "tcp" && protocol_name != "rdma") {
        throw std::invalid_argument("expected tcp or rdma transport");
    }
    const auto transport = protocol_name == "rdma" ? Stream::Kind::Rdma : Stream::Kind::Tcp;
    const auto port = parse_int(argv[4], 1, 65535);
    if (mode == "serve" && (argc == 5 || argc == 6)) {
        serve(transport, argv[3], port, use_cuda(argc == 6 ? argv[5] : "cpu"));
    } else if (mode == "prefill" && (argc == 7 || argc == 8)) {
        prefill(transport, argv[3], port, argv[5], parse_int(argv[6], 1, fixture::max_generated),
                use_cuda(argc == 8 ? argv[7] : "cpu"));
    } else {
        throw std::invalid_argument("invalid mode or argument count");
    }
} catch (const std::exception& error) {
    std::cerr << "split_inference: " << error.what() << '\n';
    return 1;
}
