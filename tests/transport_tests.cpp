#include "model.hpp"
#include "transport.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <latch>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
constexpr auto tcp = Stream::Kind::Tcp;
constexpr std::string_view loopback = "127.0.0.1";
constexpr std::uint32_t kv03 = 0x4b563033;
constexpr std::uint32_t done = 0x444f4e45;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<class Error, class Action>
void rejects(Action action, std::string_view expected) {
    try {
        action();
    } catch (const Error& error) {
        require(std::string_view(error.what()).find(expected) != std::string_view::npos, error.what());
        return;
    }
    throw std::runtime_error("unexpected success");
}

class TempDir {
public:
    TempDir() {
        auto value = (std::filesystem::temp_directory_path() / "split-test-XXXXXX").string();
        if (!mkdtemp(value.data())) throw std::runtime_error("could not create temporary directory");
        path = std::move(value);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

class Child {
public:
    Child(const char* exe, const std::filesystem::path& scratch, std::string_view name,
          std::vector<std::string> options)
        : out_path_(scratch / (std::string(name) + ".out")),
          err_path_(scratch / (std::string(name) + ".err")) {
        std::vector<char*> args{const_cast<char*>(exe)};
        for (auto& option : options) args.push_back(option.data());
        args.push_back(nullptr);
        const int out = open(out_path_.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (out == -1) throw std::runtime_error("open child output");
        const int err = open(err_path_.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (err == -1) {
            close(out);
            throw std::runtime_error("open child error log");
        }
        pid_ = fork();
        if (pid_ == 0) {
            if (dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0) _exit(126);
            close(out);
            close(err);
            execv(exe, args.data());
            _exit(127);
        }
        close(out);
        close(err);
        if (pid_ < 0) throw std::runtime_error("fork inference process");
    }

    ~Child() {
        if (pid_ > 0) {
            kill(pid_, SIGKILL);
            while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
        }
    }

    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;

    std::string stderr_text() const { return read_file(err_path_); }
    std::string stdout_text() const { return read_file(out_path_); }

    int join() {
        const auto deadline = Clock::now() + 8s;
        while (Clock::now() < deadline) {
            int status = 0;
            const auto waited = waitpid(pid_, &status, WNOHANG);
            if (waited == pid_) {
                pid_ = -1;
                return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            }
            if (waited < 0 && errno != EINTR) throw std::runtime_error("wait for process");
            std::this_thread::sleep_for(5ms);
        }
        throw std::runtime_error("process timed out: " + stderr_text());
    }

    std::uint16_t port() const {
        const auto deadline = Clock::now() + 5s;
        constexpr std::string_view marker = "listening on 127.0.0.1:";
        while (Clock::now() < deadline) {
            const auto log = stderr_text();
            if (const auto found = log.find(marker); found != std::string::npos) {
                const auto start = found + marker.size();
                if (const auto end = log.find('\n', start); end != std::string::npos) {
                    const auto port = std::stoi(log.substr(start, end - start));
                    require(port > 0 && port <= 65535, "not an assigned port");
                    return static_cast<std::uint16_t>(port);
                }
            }
            std::this_thread::sleep_for(5ms);
        }
        throw std::runtime_error("worker failed to start: " + stderr_text());
    }

private:
    std::filesystem::path out_path_, err_path_;
    int pid_ = -1;
};

void expect_exit(Child& process, int status, std::string_view expected_log) {
    const int actual = process.join();
    const auto log = process.stderr_text();
    require(actual == status && log.find(expected_log) != std::string::npos, log);
}

std::string generation(const fixture::Cache& cache, int count) {
    std::string result;
    fixture::decode_cpu(cache, count, [&](unsigned char token) {
        require(std::string_view(fixture::vocabulary).find(static_cast<char>(token)) != std::string_view::npos,
                "token outside fixture vocabulary");
        result += static_cast<char>(token);
    });
    require(result.size() == static_cast<std::size_t>(count), "missing tokens");
    return result;
}

void model_prefix() {
    const auto short_prompt = fixture::prefill_cpu("one two");
    const auto full_prompt = fixture::prefill_cpu("one two three");
    require(short_prompt.payload_bytes() == sizeof(float) *
        (2 * fixture::layers * fixture::width * short_prompt.size() + fixture::width), "payload size");
    for (int layer = 0; layer < fixture::layers; ++layer) {
        const auto& a = short_prompt.attention[layer];
        const auto& b = full_prompt.attention[layer];
        require(std::equal(a.keys.begin(), a.keys.end(), b.keys.begin()), "future keys affected prefix");
        require(std::equal(a.values.begin(), a.values.end(), b.values.begin()), "future values affected prefix");
    }
    auto first = fixture::prefill_cpu("a!");
    const auto second = fixture::prefill_cpu("b!");
    require(first.hidden != second.hidden, "last output ignored the causal context");
    const auto long_decode = generation(first, 17);
    require(long_decode.starts_with(generation(first, 1)) && long_decode == generation(first, 17),
            "greedy generation was inconsistent");
}

void tcp_transport() {
    std::uint16_t unused = 0;
    {
        auto listener = Stream::listen(tcp, loopback, 0);
        unused = listener.local_port();
        rejects<std::system_error>([&] { Stream::listen(tcp, loopback, unused); }, "bind");
        auto sender = Stream::connect(tcp, loopback, unused, 1s);
        auto peer = listener.accept(1s);
        const auto moved = std::move(sender);
        const std::array<unsigned char, 4> bytes{42, 0, 255, 7};
        moved.send(bytes.data(), 2);
        moved.send(bytes.data() + 2, 2);
        std::array<unsigned char, 4> result{};
        peer.receive(result.data(), result.size());
        require(result == bytes, "wrong data when reading a segmented write");
        rejects<std::system_error>([&] { peer.receive(result.data(), 1); }, "receive");
        rejects<std::system_error>([&] { moved.receive(result.data(), 1); }, "receive");
    }
    rejects<std::system_error>([&] { Stream::connect(tcp, loopback, unused); }, "connect");
}

void interrupted_transport() {
    struct SignalGuard {
        struct sigaction previous{};
        SignalGuard() {
            struct sigaction handler{};
            handler.sa_handler = [](int) {};
            sigemptyset(&handler.sa_mask);
            require(sigaction(SIGUSR1, &handler, &previous) == 0, "install signal handler");
        }
        ~SignalGuard() { sigaction(SIGUSR1, &previous, nullptr); }
    } signals;
    const auto interrupt = [](std::jthread& thread) {
        for (int i = 0; i < 8; ++i) {
            std::this_thread::sleep_for(3ms);
            require(pthread_kill(thread.native_handle(), SIGUSR1) == 0, "interrupt blocked thread");
        }
    };
    auto listener = Stream::listen(tcp, loopback, 0);
    auto sender = Stream::connect(tcp, loopback, listener.local_port());
    auto peer = listener.accept(2s);
    unsigned char received = 0;
    std::exception_ptr error;
    std::latch started(1);
    std::jthread receiver([&] {
        started.count_down();
        try { peer.receive(&received, 1); }
        catch (...) { error = std::current_exception(); }
    });
    started.wait();
    interrupt(receiver);
    const unsigned char expected = 42;
    sender.send(&expected, 1);
    receiver.join();
    if (error) std::rethrow_exception(error);
    require(received == expected, "interrupted receive lost its byte");

    // The payload exceeds the socket buffers, so a blocked writer must resume
    // after both signal delivery and partial progress on the socket.
    std::vector<unsigned char> output(8 * 1024 * 1024, expected), echoed(output.size());
    std::latch writing(1);
    std::jthread writer([&] {
        writing.count_down();
        try { sender.send(output.data(), output.size()); }
        catch (...) { error = std::current_exception(); }
    });
    writing.wait();
    interrupt(writer);
    peer.receive(echoed.data(), echoed.size());
    writer.join();
    if (error) std::rethrow_exception(error);
    require(echoed == output, "interrupted send lost or duplicated bytes");
}

void reject_packet(std::uint16_t port, std::uint32_t protocol, bool bad_payload) {
    auto client = Stream::connect(tcp, loopback, port);
    const std::array header{htonl(protocol), htonl(1U), htonl(2U)};
    client.send(header.data(), sizeof(header));
    if (bad_payload) {
        std::array<std::uint32_t, (2 * fixture::layers + 1) * fixture::width> data{};
        data[0] = htonl(std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity()));
        client.send(data.data(), sizeof(data));
    }
    unsigned char reply = 0;
    rejects<std::runtime_error>([&] { client.receive(&reply, 1); }, "stream closed");
}

void protocol(const char* exe, const std::filesystem::path& scratch) {
    Child server(exe, scratch, "loop", {"serve-loop", "tcp", std::string(loopback), "0", "2", "cpu", "6"});
    const auto port = server.port();
    const auto destination = std::to_string(port);
    reject_packet(port, kv03 - 1, false);
    reject_packet(port, kv03, true);
    { const auto closed = Stream::connect(tcp, loopback, port); }

    Child comparison(exe, scratch, "compare",
        {"compare", "tcp", std::string(loopback), destination, "actual causal attention", "9", "1", "cpu"});
    expect_exit(comparison, 0, "parity=exact pairs=1");
    const auto csv = comparison.stdout_text();
    require(csv.find("1,monolithic,cpu,23,9,0,") != std::string::npos, "missing local sample");
    require(csv.find("1,split,cpu,23,9,23808,") != std::string::npos, "missing split sample");

    Child client(exe, scratch, "stream", {"prefill", "tcp", std::string(loopback), destination, "a!", "17"});
    expect_exit(client, 0, "state_bytes=2304");
    require(client.stdout_text() == generation(fixture::prefill_cpu("a!"), 17) + "\n", "streamed wrong tokens");
    expect_exit(server, 0, "decode_tokens=17");
    const auto log = server.stderr_text();
    require(log.find("incompatible KV handoff header") != std::string::npos &&
            log.find("invalid fixture state entry") != std::string::npos &&
            log.find("stream closed") != std::string::npos, "bad input did not reach the worker");

    Child once(exe, scratch, "once", {"serve", "tcp", std::string(loopback), "0"});
    const auto once_port = std::to_string(once.port());
    Child only(exe, scratch, "only", {"prefill", "tcp", std::string(loopback), once_port, "x", "1"});
    expect_exit(only, 0, "ttft_ms=");
    expect_exit(once, 0, "decode_tokens=1");
}

void parity_rejects(const char* exe, const std::filesystem::path& scratch) {
    auto listener = Stream::listen(tcp, loopback, 0);
    const auto port = std::to_string(listener.local_port());
    Child comparison(exe, scratch, "wrong",
        {"compare", "tcp", std::string(loopback), port, "x", "1", "1", "cpu"});
    auto peer = listener.accept(1s);
    std::array<std::uint32_t, 3 + (2 * fixture::layers + 1) * fixture::width> request{};
    peer.receive(request.data(), sizeof(request));
    const auto expected = generation(fixture::prefill_cpu("x"), 1);
    const unsigned char wrong = expected.front() == 'a' ? 'b' : 'a';
    peer.send(&wrong, 1);
    const std::array completion{htonl(done), htonl(1)};
    peer.send(completion.data(), sizeof(completion));
    expect_exit(comparison, 1, "split output differs from monolithic at token 0");
}

void completion_required(const char* exe, const std::filesystem::path& scratch) {
    auto listener = Stream::listen(tcp, loopback, 0);
    const auto port = std::to_string(listener.local_port());
    const auto simulate = [&](std::string label, const std::vector<std::uint32_t>& trailer,
                              std::string_view error, bool invalid_token = false) {
        Child client(exe, scratch, std::move(label),
            {"prefill", "tcp", std::string(loopback), port, "x", "1"});
        {
            auto peer = listener.accept(1s);
            std::array<std::uint32_t, 3 + (2 * fixture::layers + 1) * fixture::width> request{};
            peer.receive(request.data(), sizeof(request));
            require(ntohl(request[0]) == kv03, "client did not request version KV03");
            auto frame = invalid_token ? std::string("!") : generation(fixture::prefill_cpu("x"), 1);
            if (!trailer.empty()) {
                frame.append(reinterpret_cast<const char*>(trailer.data()), trailer.size() * sizeof(trailer[0]));
            }
            peer.send(frame.data(), frame.size());
        }
        expect_exit(client, 1, error);
        require(client.stdout_text().size() == (invalid_token ? 0 : 1),
                "invalid token was printed or incomplete stream reported success");
        require(client.stderr_text().find("total_ms=") == std::string::npos,
                "incomplete stream was included in timing");
    };
    simulate("no-completion", {}, "decode completion was not received");
    simulate("half-completion", {htonl(done)}, "decode completion was not received");
    simulate("wrong-count", {htonl(done), htonl(0)}, "invalid decode completion");
    simulate("wrong-marker", {htonl(kv03), htonl(1)}, "invalid decode completion");
    simulate("invalid-token", {htonl(done), htonl(1)}, "invalid decode token", true);
}

}  // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "provide the inference executable path");
    const TempDir temp;
    model_prefix();
    tcp_transport();
    interrupted_transport();
    protocol(argv[1], temp.path);
    parity_rejects(argv[1], temp.path);
    completion_required(argv[1], temp.path);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
