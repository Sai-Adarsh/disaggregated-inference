#pragma once

#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

namespace fixture {

constexpr int width = 64;
constexpr int max_prompt = 8192;
constexpr int max_generated = 128;
constexpr char vocabulary[] = " abcdefghijklmnopqrstuvwxyz";
constexpr int vocabulary_size = sizeof(vocabulary) - 1;

struct Cache {
    std::vector<float> keys;
    std::vector<float> values;
    unsigned char last = 0;

    std::size_t size() const { return keys.size() / width; }
};

using Emit = std::function<void(unsigned char)>;

Cache prefill_cpu(std::string_view prompt);
void decode_cpu(Cache cache, int steps, const Emit& emit);

#ifdef INFERENCE_CUDA
Cache prefill_cuda(std::string_view prompt);
void decode_cuda(const Cache& cache, int steps, const Emit& emit);
#endif

}  // namespace fixture
