#include "model.hpp"
#include "feature.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace fixture {
namespace {

void check(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

template <class Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(&data_, count * sizeof(Value))); }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Value* get() const { return data_; }

private:
    Value* data_ = nullptr;
};

__global__ void build_prompt(const unsigned char* prompt, float* keys, float* values) {
    const int channel = threadIdx.x;
    const int index = blockIdx.x * width + channel;
    const unsigned char token = prompt[blockIdx.x];
    keys[index] = feature(token, channel, key_salt);
    values[index] = feature(token, channel, value_salt);
}

__global__ void append_token(unsigned char token, float* keys, float* values, int row) {
    const int channel = threadIdx.x;
    const int index = row * width + channel;
    keys[index] = feature(token, channel, key_salt);
    values[index] = feature(token, channel, value_salt);
}

__global__ void score_keys(const float* keys, float* mass, unsigned char last) {
    __shared__ float products[width];
    const int channel = threadIdx.x;
    products[channel] = feature(last, channel, query_salt) * keys[blockIdx.x * width + channel];
    __syncthreads();
    for (int stride = width / 2; stride > 0; stride /= 2) {
        if (channel < stride) {
            products[channel] += products[channel + stride];
        }
        __syncthreads();
    }
    if (channel == 0) {
        // Fixture inputs keep the scaled dot product inside [-8, 8].
        mass[blockIdx.x] = expf(products[0] * attention_scale);
    }
}

constexpr int tile_rows = 16;

__global__ void weighted_values(const float* values, const float* mass, float* context, int rows) {
    __shared__ float partial[tile_rows][width];
    const int channel = threadIdx.x;
    const int lane_row = threadIdx.y;
    float total = 0;
    for (int row = lane_row; row < rows; row += tile_rows) {
        total += mass[row] * values[row * width + channel];
    }
    partial[lane_row][channel] = total;
    __syncthreads();
    for (int stride = tile_rows / 2; stride > 0; stride /= 2) {
        if (lane_row < stride) {
            partial[lane_row][channel] += partial[lane_row + stride][channel];
        }
        __syncthreads();
    }
    if (lane_row == 0) {
        // Dropping the common positive normalizer preserves the greedy argmax.
        context[channel] = partial[0][channel];
    }
}

__global__ void choose_token(const float* context, unsigned char* output) {
    __shared__ float scores[vocabulary_size];
    const int candidate = threadIdx.x;
    const unsigned char token = candidate == 0 ? ' ' : 'a' + candidate - 1;
    float score = 0;
    for (int i = 0; i < width; ++i) {
        score += context[i] * feature(token, i, output_salt);
    }
    scores[candidate] = score;
    __syncthreads();
    if (candidate == 0) {
        int best = 0;
        for (int i = 1; i < vocabulary_size; ++i) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        *output = best == 0 ? ' ' : 'a' + best - 1;
    }
}

}  // namespace

Cache prefill_cuda(std::string_view prompt) {
    const std::size_t elements = prompt.size() * width;
    DeviceBuffer<unsigned char> tokens(prompt.size());
    DeviceBuffer<float> keys(elements);
    DeviceBuffer<float> values(elements);
    check(cudaMemcpy(tokens.get(), prompt.data(), prompt.size(), cudaMemcpyHostToDevice));
    build_prompt<<<prompt.size(), width>>>(tokens.get(), keys.get(), values.get());
    check(cudaGetLastError());
    Cache cache{std::vector<float>(elements), std::vector<float>(elements),
                static_cast<unsigned char>(prompt.back())};
    check(cudaMemcpy(cache.keys.data(), keys.get(), elements * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(cache.values.data(), values.get(), elements * sizeof(float), cudaMemcpyDeviceToHost));
    return cache;
}

void decode_cuda(const Cache& cache, int steps, const Emit& emit) {
    int rows = static_cast<int>(cache.size());
    const auto capacity = static_cast<std::size_t>(rows + steps);
    DeviceBuffer<float> keys(capacity * width);
    DeviceBuffer<float> values(capacity * width);
    DeviceBuffer<float> mass(capacity);
    DeviceBuffer<float> context(width);
    DeviceBuffer<unsigned char> output(1);
    const auto bytes = cache.keys.size() * sizeof(float);
    check(cudaMemcpy(keys.get(), cache.keys.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(values.get(), cache.values.data(), bytes, cudaMemcpyHostToDevice));

    unsigned char last = cache.last;
    for (int step = 0; step < steps; ++step) {
        score_keys<<<rows, width>>>(keys.get(), mass.get(), last);
        check(cudaGetLastError());
        weighted_values<<<1, dim3(width, tile_rows)>>>(values.get(), mass.get(), context.get(), rows);
        check(cudaGetLastError());
        choose_token<<<1, vocabulary_size>>>(context.get(), output.get());
        check(cudaGetLastError());
        check(cudaMemcpy(&last, output.get(), sizeof(last), cudaMemcpyDeviceToHost));
        append_token<<<1, width>>>(last, keys.get(), values.get(), rows++);
        check(cudaGetLastError());
        emit(last);
    }
    check(cudaDeviceSynchronize());
}

}  // namespace fixture
