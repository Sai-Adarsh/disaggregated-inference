#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

namespace fixture {

constexpr int width = 64;
constexpr int heads = 4;
constexpr int head_width = width / heads;
constexpr int layers = 2;
constexpr int max_prompt = 8192;
constexpr int max_generated = 128;
constexpr char vocabulary[] = " abcdefghijklmnopqrstuvwxyz";
constexpr int vocabulary_size = sizeof(vocabulary) - 1;

struct LayerCache {
    std::vector<float> keys;
    std::vector<float> values;
};

struct Cache {
    std::array<LayerCache, layers> attention;
    std::array<float, width> hidden{};

    std::size_t size() const { return attention[0].keys.size() / width; }
    std::size_t payload_bytes() const { return sizeof(float) * (2 * layers * size() * width + width); }
};

using Emit = std::function<void(unsigned char)>;

Cache prefill_cpu(std::string_view prompt);
void decode_cpu(Cache cache, int steps, const Emit& emit);

#ifdef INFERENCE_CUDA
Cache prefill_cuda(std::string_view prompt);
void decode_cuda(const Cache& cache, int steps, const Emit& emit);
void monolithic_cuda(std::string_view prompt, int steps, const Emit& emit,
                     const std::function<void()>& prefilled);
#endif

}  // namespace fixture
