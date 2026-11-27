#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

class Stream {
public:
    enum class Kind { Tcp, Rdma };

    static Stream listen(Kind kind, std::string_view address, std::uint16_t port);
    static Stream connect(Kind kind, std::string_view address, std::uint16_t port);
    Stream accept() const;
    void send(const void* data, std::size_t bytes) const;
    void receive(void* data, std::size_t bytes) const;

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream();

private:
    Stream(Kind kind, int descriptor) : kind_(kind), descriptor_(descriptor) {}
    static int open(Kind kind);

    Kind kind_;
    int descriptor_;
};
