#include "transport.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef INFERENCE_RDMA
#include <rdma/rsocket.h>
#endif

#include <algorithm>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace {

void check(ssize_t result, const char* operation) {
    if (result < 0) {
        throw std::system_error(errno, std::generic_category(), operation);
    }
}

void configure_tcp(int fd, std::chrono::seconds idle_timeout) {
    const int on = 1;
    check(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on)), "setsockopt");
#ifdef SO_NOSIGPIPE
    check(setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on)), "suppress TCP SIGPIPE");
#endif
    if (idle_timeout.count() <= 0) return;
    timeval timeout{};
    timeout.tv_sec = idle_timeout.count();
    check(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), "set TCP receive timeout");
    check(setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)), "set TCP send timeout");
}

void connect_tcp(int fd, const sockaddr* target, socklen_t length, std::chrono::seconds timeout) {
    if (timeout.count() <= 0) {
        check(::connect(fd, target, length), "connect");
        return;
    }

    const int flags = fcntl(fd, F_GETFL);
    check(flags, "read TCP socket flags");
    check(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "set TCP socket nonblocking");
    const auto started = std::chrono::steady_clock::now();
    if (::connect(fd, target, length) < 0) {
        if (errno != EINPROGRESS && errno != EINTR) check(-1, "connect");

        using Milliseconds = std::chrono::duration<double, std::milli>;
        constexpr Milliseconds max_poll_wait{std::numeric_limits<int>::max()};
        pollfd socket{fd, POLLOUT, 0};
        for (;;) {
            const auto elapsed = std::chrono::steady_clock::now() - started;
            const auto remaining = Milliseconds{timeout} - Milliseconds{elapsed};
            if (remaining <= Milliseconds::zero()) {
                throw std::system_error(std::make_error_code(std::errc::timed_out), "TCP connect");
            }
            const auto wait = std::chrono::ceil<std::chrono::milliseconds>(std::min(remaining, max_poll_wait));
            const int ready = poll(&socket, 1, static_cast<int>(wait.count()));
            if (ready == 0 || (ready < 0 && errno == EINTR)) continue;
            check(ready, "poll TCP connect");
            break;
        }
        int error = 0;
        socklen_t size = sizeof(error);
        check(getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size), "check TCP connect");
        if (error != 0) throw std::system_error(error, std::generic_category(), "connect");
    }
    check(fcntl(fd, F_SETFL, flags), "restore TCP socket flags");
}

sockaddr_in endpoint(std::string_view address, std::uint16_t port) {
    sockaddr_in host{};
    host.sin_family = AF_INET;
    host.sin_port = htons(port);
    if (inet_pton(AF_INET, std::string(address).c_str(), &host.sin_addr) != 1) {
        throw std::invalid_argument("expected an IPv4 address");
    }
    return host;
}

}  // namespace

int Stream::open(Kind kind) {
    if (kind == Kind::Rdma) {
#ifdef INFERENCE_RDMA
        const int fd = rsocket(AF_INET, SOCK_STREAM, 0);
        check(fd, "rsocket");
        return fd;
#else
        throw std::invalid_argument("rebuild with ENABLE_RDMA=ON to use RDMA");
#endif
    }
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    check(fd, "socket");
    return fd;
}

Stream::Stream(Stream&& other) noexcept
    : kind_(other.kind_), descriptor_(std::exchange(other.descriptor_, -1)) {}

Stream::~Stream() {
    if (descriptor_ < 0) {
        return;
    }
#ifdef INFERENCE_RDMA
    if (kind_ == Kind::Rdma) {
        rclose(descriptor_);
        return;
    }
#endif
    close(descriptor_);
}

Stream Stream::listen(Kind kind, std::string_view address, std::uint16_t port) {
    const auto target = endpoint(address, port);
    const auto* target_address = reinterpret_cast<const sockaddr*>(&target);
    const int fd = open(kind);
    try {
#ifdef INFERENCE_RDMA
        if (kind == Kind::Rdma) {
            check(rbind(fd, target_address, sizeof(target)), "rbind");
            check(rlisten(fd, 8), "rlisten");
            return Stream(kind, fd);
        }
#endif
        const int on = 1;
        check(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)), "setsockopt");
        check(bind(fd, target_address, sizeof(target)), "bind");
        check(::listen(fd, 8), "listen");
        return Stream(kind, fd);
    } catch (...) {
        const Stream cleanup(kind, fd);
        throw;
    }
}

Stream Stream::connect(Kind kind, std::string_view address, std::uint16_t port,
                       std::chrono::seconds tcp_timeout) {
    const auto target = endpoint(address, port);
    const auto* target_address = reinterpret_cast<const sockaddr*>(&target);
    const int fd = open(kind);
    try {
#ifdef INFERENCE_RDMA
        if (kind == Kind::Rdma) {
            check(rconnect(fd, target_address, sizeof(target)), "rconnect");
        } else
#endif
        {
            connect_tcp(fd, target_address, sizeof(target), tcp_timeout);
            configure_tcp(fd, tcp_timeout);
        }
    } catch (...) {
        const Stream cleanup(kind, fd);
        throw;
    }
    return Stream(kind, fd);
}

Stream Stream::accept(std::chrono::seconds tcp_idle_timeout) const {
    int fd = -1;
    do {
#ifdef INFERENCE_RDMA
        if (kind_ == Kind::Rdma) {
            fd = raccept(descriptor_, nullptr, nullptr);
        } else
#endif
        {
            fd = ::accept(descriptor_, nullptr, nullptr);
        }
    } while (fd < 0 && errno == EINTR);
    check(fd, "accept");
    try {
        if (kind_ == Kind::Tcp) {
            configure_tcp(fd, tcp_idle_timeout);
        }
    } catch (...) {
        const Stream cleanup(kind_, fd);
        throw;
    }
    return Stream(kind_, fd);
}

std::uint16_t Stream::local_port() const {
    sockaddr_in address{};
    auto size = static_cast<socklen_t>(sizeof(address));
    auto* name = reinterpret_cast<sockaddr*>(&address);
#ifdef INFERENCE_RDMA
    if (kind_ == Kind::Rdma) {
        check(rgetsockname(descriptor_, name, &size), "rgetsockname");
    } else
#endif
    {
        check(getsockname(descriptor_, name, &size), "getsockname");
    }
    return ntohs(address.sin_port);
}

void Stream::send(const void* data, std::size_t remaining) const {
    const auto* current = static_cast<const unsigned char*>(data);
    while (remaining != 0) {
        ssize_t bytes = 0;
#ifdef INFERENCE_RDMA
        if (kind_ == Kind::Rdma) {
            bytes = rsend(descriptor_, current, remaining, 0);
        } else
#endif
        {
#ifdef SO_NOSIGPIPE
            constexpr int flags = 0;  // configure_tcp suppresses SIGPIPE on this socket.
#else
            constexpr int flags = MSG_NOSIGNAL;
#endif
            bytes = ::send(descriptor_, current, remaining, flags);
        }
        if (bytes < 0 && errno == EINTR) {
            continue;
        }
        check(bytes, "send");
        if (bytes == 0) {
            throw std::runtime_error("stream closed during send");
        }
        current += bytes;
        remaining -= static_cast<std::size_t>(bytes);
    }
}

void Stream::receive(void* data, std::size_t remaining) const {
    auto* current = static_cast<unsigned char*>(data);
    while (remaining != 0) {
        ssize_t bytes = 0;
#ifdef INFERENCE_RDMA
        if (kind_ == Kind::Rdma) {
            bytes = rrecv(descriptor_, current, remaining, 0);
        } else
#endif
        {
            bytes = recv(descriptor_, current, remaining, 0);
        }
        if (bytes < 0 && errno == EINTR) {
            continue;
        }
        check(bytes, "receive");
        if (bytes == 0) {
            throw std::runtime_error("stream closed before the message was complete");
        }
        current += bytes;
        remaining -= static_cast<std::size_t>(bytes);
    }
}
