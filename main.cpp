#include "model.hpp"
#include "transport.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t protocol = 0x4b563033;  // KV03, confirms decode completion
constexpr std::uint32_t completion_magic = 0x444f4e45;  // DONE
constexpr int max_decode_workers = 64;
constexpr std::chrono::seconds tcp_idle_timeout{30};
std::mutex decode_log_mutex;

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
                            htonl(steps)};
    stream.send(header.data(), sizeof(header));
    std::vector<std::uint32_t> payload;
    payload.reserve(cache.payload_bytes() / sizeof(float));
    const auto append = [&payload](const auto& values) {
        for (const float value : values) payload.push_back(htonl(std::bit_cast<std::uint32_t>(value)));
    };
    for (const auto& layer : cache.attention) {
        append(layer.keys);
        append(layer.values);
    }
    append(cache.hidden);
    stream.send(payload.data(), payload.size() * sizeof(payload[0]));
}

fixture::Cache read_cache(const Stream& stream, int& steps) {
    std::array<std::uint32_t, 3> header{};
    stream.receive(header.data(), sizeof(header));
    const auto rows = ntohl(header[1]);
    const auto requested_steps = ntohl(header[2]);
    if (ntohl(header[0]) != protocol || rows == 0 || rows > fixture::max_prompt
        || requested_steps == 0 || requested_steps > fixture::max_generated) {
        throw std::runtime_error("incompatible KV handoff header");
    }
    steps = static_cast<int>(requested_steps);
    const auto matrix_elements = std::size_t{rows} * fixture::width;
    std::vector<std::uint32_t> payload(2 * fixture::layers * matrix_elements + fixture::width);
    stream.receive(payload.data(), payload.size() * sizeof(payload[0]));
    fixture::Cache cache;
    auto cursor = payload.cbegin();
    const auto read = [&cursor](auto& values) {
        for (auto& value : values) {
            value = std::bit_cast<float>(ntohl(*cursor++));
            if (!std::isfinite(value) || std::abs(value) > 1.0e6F) {
                throw std::runtime_error("invalid fixture state entry");
            }
        }
    };
    for (auto& layer : cache.attention) {
        layer.keys.resize(matrix_elements);
        layer.values.resize(matrix_elements);
        read(layer.keys);
        read(layer.values);
    }
    read(cache.hidden);
    return cache;
}

void decode_request(const Stream& client, bool gpu) {
    const auto start = Clock::now();
    int steps = 0;
    auto cache = read_cache(client, steps);
    const auto received = Clock::now();
    std::uint32_t emitted = 0;
    const auto emit = [&client, &emitted](unsigned char token) {
        client.send(&token, 1);
        ++emitted;
    };
#ifdef INFERENCE_CUDA
    if (gpu) {
        fixture::decode_cuda(cache, steps, emit);
    } else
#endif
    {
        fixture::decode_cpu(std::move(cache), steps, emit);
    }
    const auto finished = Clock::now();
    const std::array completion{htonl(completion_magic), htonl(emitted)};
    client.send(completion.data(), sizeof(completion));
    const std::lock_guard lock(decode_log_mutex);
    std::cerr << "backend=" << (gpu ? "cuda" : "cpu")
              << " decode_tokens=" << steps << " receive_ms=" << milliseconds(received - start)
              << " decode_ms=" << milliseconds(finished - received) << '\n';
}

class DecodeWorkers {
public:
    DecodeWorkers(int capacity, bool gpu) : capacity_(capacity), gpu_(gpu) {
        workers_.reserve(capacity_);
        for (int i = 0; i < capacity_; ++i) {
            workers_.emplace_back([this](std::stop_token stop) { run(stop); });
        }
    }

    void accept_connections(const Stream& listener, int request_limit) {
        for (int accepted = 0; request_limit == 0 || accepted < request_limit;) {
            std::unique_lock lock(mutex_);
            slot_available_.wait(lock, [this] { return in_flight_ < capacity_; });
            lock.unlock();

            auto client = listener.accept(tcp_idle_timeout);
            lock.lock();
            pending_.push_back(std::move(client));
            ++in_flight_;
            if (request_limit != 0) ++accepted;
            lock.unlock();
            work_available_.notify_one();
        }
    }

private:
    void run(std::stop_token stop) {
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                if (!work_available_.wait(lock, stop, [this] { return !pending_.empty(); })) {
                    return;
                }
                auto client = std::move(pending_.front());
                pending_.pop_front();
                lock.unlock();
                try {
                    decode_request(client, gpu_);
                } catch (const std::exception& error) {
                    const std::lock_guard log_lock(decode_log_mutex);
                    std::cerr << "decode request failed: " << error.what() << '\n';
                }
            }
            {
                const std::lock_guard lock(mutex_);
                --in_flight_;
            }
            slot_available_.notify_one();
        }
    }

    int capacity_;
    bool gpu_;
    int in_flight_ = 0;
    std::mutex mutex_;
    std::condition_variable slot_available_;
    std::condition_variable_any work_available_;
    std::deque<Stream> pending_;
    // Worker threads join before the synchronization state is destroyed.
    std::vector<std::jthread> workers_;
};

void serve(Stream::Kind transport, std::string_view address, int port, bool gpu,
           int workers = 0, int request_limit = 0) {
    const auto listener = Stream::listen(transport, address, port);
    const auto bound_port = listener.local_port();
    if (workers == 0) {
        std::cerr << "listening on " << address << ':' << bound_port << '\n';
        decode_request(listener.accept(tcp_idle_timeout), gpu);
        return;
    }

    DecodeWorkers pool(workers, gpu);
    std::cerr << "listening on " << address << ':' << bound_port << " workers=" << workers << '\n';
    pool.accept_connections(listener, request_limit);
}

struct Measurement {
    std::string tokens;
    std::size_t payload_bytes = 0;
    double prefill_ms = 0;
    double connect_send_ms = 0;
    double ttft_ms = 0;
    double total_ms = 0;
};

void check_prompt(std::string_view prompt) {
    if (prompt.empty() || prompt.size() > fixture::max_prompt) {
        throw std::invalid_argument("prompt must contain 1–8192 bytes");
    }
}

Measurement split(Stream::Kind transport, std::string_view address, int port,
                  std::string_view prompt, int steps, bool gpu, bool print_tokens) {
    Measurement measured;
    measured.tokens.reserve(steps);

    const auto start = Clock::now();
    const auto cache =
#ifdef INFERENCE_CUDA
        gpu ? fixture::prefill_cuda(prompt) :
#endif
        fixture::prefill_cpu(prompt);
    const auto computed = Clock::now();
    measured.payload_bytes = cache.payload_bytes();
    measured.prefill_ms = milliseconds(computed - start);
    const auto stream = Stream::connect(transport, address, port, tcp_idle_timeout);
    write_cache(stream, cache, steps);
    measured.connect_send_ms = milliseconds(Clock::now() - computed);
    for (int i = 0; i < steps; ++i) {
        unsigned char next = 0;
        stream.receive(&next, 1);
        if (i == 0) measured.ttft_ms = milliseconds(Clock::now() - start);
        measured.tokens.push_back(static_cast<char>(next));
        if (print_tokens) std::cout << next << std::flush;
    }
    std::array<std::uint32_t, 2> completion{};
    try {
        stream.receive(completion.data(), sizeof(completion));
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(std::string("decode completion was not received: ") + error.what());
    }
    if (ntohl(completion[0]) != completion_magic || ntohl(completion[1]) != measured.tokens.size()) {
        throw std::runtime_error("invalid decode completion");
    }
    measured.total_ms = milliseconds(Clock::now() - start);
    return measured;
}

Measurement local(std::string_view prompt, int steps, bool gpu) {
    Measurement measured;
    measured.tokens.reserve(steps);
    const auto start = Clock::now();
    const auto emit = [&](unsigned char next) {
        if (measured.tokens.empty()) measured.ttft_ms = milliseconds(Clock::now() - start);
        measured.tokens.push_back(static_cast<char>(next));
    };
    const auto prefilled = [&] { measured.prefill_ms = milliseconds(Clock::now() - start); };
#ifdef INFERENCE_CUDA
    if (gpu) {
        fixture::monolithic_cuda(prompt, steps, emit, prefilled);
    } else
#endif
    {
        auto cache = fixture::prefill_cpu(prompt);
        prefilled();
        fixture::decode_cpu(std::move(cache), steps, emit);
    }
    measured.total_ms = milliseconds(Clock::now() - start);
    return measured;
}

void prefill(Stream::Kind transport, std::string_view address, int port,
             std::string_view prompt, int steps, bool gpu) {
    check_prompt(prompt);
    const auto measured = split(transport, address, port, prompt, steps, gpu, true);
    std::cout << '\n';
    std::cerr << "backend=" << (gpu ? "cuda" : "cpu")
              << " prompt_bytes=" << prompt.size() << " state_bytes=" << measured.payload_bytes
              << " prefill_ms=" << measured.prefill_ms << " connect_send_ms=" << measured.connect_send_ms
              << " ttft_ms=" << measured.ttft_ms << " total_ms=" << measured.total_ms << '\n';
}

void compare(Stream::Kind transport, std::string_view address, int port,
             std::string_view prompt, int steps, int trials, bool gpu) {
    check_prompt(prompt);
    std::cout << "trial,path,prefill_backend,prompt_tokens,output_tokens,state_bytes,prefill_ms,connect_send_ms,ttft_ms,total_ms\n";
    for (int trial = 0; trial <= trials; ++trial) {
        Measurement monolithic, disaggregated;
        // Exclude one warmup pair and rotate the order of subsequent pairs.
        if (trial % 2 == 0) {
            monolithic = local(prompt, steps, gpu);
            disaggregated = split(transport, address, port, prompt, steps, gpu, false);
        } else {
            disaggregated = split(transport, address, port, prompt, steps, gpu, false);
            monolithic = local(prompt, steps, gpu);
        }
        if (monolithic.tokens != disaggregated.tokens) {
            const auto mismatch = std::mismatch(monolithic.tokens.begin(), monolithic.tokens.end(),
                                               disaggregated.tokens.begin(), disaggregated.tokens.end());
            throw std::runtime_error("split output differs from monolithic at token "
                + std::to_string(mismatch.first - monolithic.tokens.begin()));
        }
        if (trial == 0) continue;
        const auto print = [&](std::string_view path, const Measurement& m) {
            std::cout << trial << ',' << path << ',' << (gpu ? "cuda" : "cpu") << ',' << prompt.size()
                      << ',' << m.tokens.size() << ',' << m.payload_bytes << ',' << m.prefill_ms << ','
                      << m.connect_send_ms << ',' << m.ttft_ms << ',' << m.total_ms << '\n';
        };
        print("monolithic", monolithic);
        print("split", disaggregated);
    }
    std::cerr << "parity=exact pairs=" << trials << " warmup_pairs=1 tokens_per_pair=" << steps << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    std::signal(SIGPIPE, SIG_IGN);
    if (argc < 5) {
        throw std::invalid_argument("usage: split_inference serve tcp|rdma IP PORT [cpu|cuda]\n"
                                    "       split_inference serve-loop tcp|rdma IP PORT WORKERS [cpu|cuda] [CONNECTIONS]\n"
                                    "       split_inference prefill tcp|rdma IP PORT PROMPT STEPS [cpu|cuda]\n"
                                    "       split_inference compare tcp|rdma IP PORT PROMPT STEPS TRIALS [cpu|cuda]");
    }
    const std::string_view mode = argv[1];
    const std::string_view protocol_name = argv[2];
    if (protocol_name != "tcp" && protocol_name != "rdma") {
        throw std::invalid_argument("expected tcp or rdma transport");
    }
    const auto transport = protocol_name == "rdma" ? Stream::Kind::Rdma : Stream::Kind::Tcp;
    const auto port = parse_int(argv[4], mode == "serve" || mode == "serve-loop" ? 0 : 1, 65535);
    if (mode == "serve" && (argc == 5 || argc == 6)) {
        serve(transport, argv[3], port, use_cuda(argc == 6 ? argv[5] : "cpu"));
    } else if (mode == "serve-loop" && argc >= 6 && argc <= 8) {
        const int workers = parse_int(argv[5], 1, max_decode_workers);
        const int request_limit = argc == 8 ? parse_int(argv[7], 1, 1000000) : 0;
        serve(transport, argv[3], port, use_cuda(argc >= 7 ? argv[6] : "cpu"), workers, request_limit);
    } else if (mode == "prefill" && (argc == 7 || argc == 8)) {
        prefill(transport, argv[3], port, argv[5], parse_int(argv[6], 1, fixture::max_generated),
                use_cuda(argc == 8 ? argv[7] : "cpu"));
    } else if (mode == "compare" && (argc == 8 || argc == 9)) {
        compare(transport, argv[3], port, argv[5], parse_int(argv[6], 1, fixture::max_generated),
                parse_int(argv[7], 1, 1000), use_cuda(argc == 9 ? argv[8] : "cpu"));
    } else {
        throw std::invalid_argument("invalid mode or argument count");
    }
} catch (const std::exception& error) {
    std::cerr << "split_inference: " << error.what() << '\n';
    return 1;
}
