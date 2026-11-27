#pragma once

namespace fixture {

#ifdef __CUDACC__
__host__ __device__
#endif
inline float feature(unsigned char token, int channel, unsigned salt) {
    unsigned bits = (static_cast<unsigned>(token) + 1) * 0x9e3779b9u;
    bits ^= (static_cast<unsigned>(channel) + 1) * 0x85ebca6bu;
    bits ^= salt;
    bits ^= bits >> 16;
    bits *= 0x7feb352du;
    bits ^= bits >> 15;
    return static_cast<float>(bits & 0xffffu) / 32768.0F - 1.0F;
}

constexpr unsigned key_salt = 0x0463e175u;
constexpr unsigned value_salt = 0x5285dcc3u;
constexpr unsigned query_salt = 0x942c5a47u;
constexpr unsigned output_salt = 0x119de1f3u;
constexpr float attention_scale = 0.125F;

}  // namespace fixture
