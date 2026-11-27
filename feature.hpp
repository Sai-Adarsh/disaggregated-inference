#pragma once

#include "model.hpp"

namespace fixture {

#ifdef __CUDACC__
#define FIXTURE_HD __host__ __device__
#else
#define FIXTURE_HD
#endif

FIXTURE_HD inline float feature(unsigned char token, int channel, unsigned salt) {
    unsigned bits = (static_cast<unsigned>(token) + 1) * 0x9e3779b9u;
    bits ^= (static_cast<unsigned>(channel) + 1) * 0x85ebca6bu;
    bits ^= salt;
    bits ^= bits >> 16;
    bits *= 0x7feb352du;
    bits ^= bits >> 15;
    return static_cast<float>(bits & 0xffffu) / 32768.0F - 1.0F;
}

FIXTURE_HD inline float embedding(unsigned char token, int position, int channel) {
    return feature(token, channel, 0x6170d719u)
        + 0.125F * feature(static_cast<unsigned char>(position), channel, 0x88a7bfbau)
        + 0.0625F * feature(static_cast<unsigned char>(position >> 8), channel, 0x1c3ef5fbu);
}

FIXTURE_HD inline unsigned char vocabulary_byte(int index) {
    return index == 0 ? ' ' : static_cast<unsigned char>('a' + index - 1);
}

#undef FIXTURE_HD

constexpr unsigned output_salt = 0x119de1f3u;
constexpr float attention_scale = 0.25F;  // 1 / sqrt(16)
constexpr int matrix_elements = width * width;
using Parameters = std::array<float, layers * 4 * matrix_elements>;
const Parameters& parameters();

}  // namespace fixture
