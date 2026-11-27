#include "feature.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdexcept>
#include <utility>

namespace fixture {
namespace {

constexpr int attention_threads = 128;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
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

__global__ void embed_prompt(const unsigned char* prompt, float* output) {
    const int channel = threadIdx.x;
    output[blockIdx.x * width + channel] = embedding(prompt[blockIdx.x], blockIdx.x, channel);
}

__global__ void embed_token(unsigned char token, int position, float* output) {
    output[threadIdx.x] = embedding(token, position, threadIdx.x);
}

__global__ void project(const float* matrix, const float* hidden, float* queries,
                        float* keys, float* values, int first_position) {
    const int col = threadIdx.x;
    const int row = blockIdx.x;
    const int offset = col * width;
    const auto* input = hidden + row * width;
    float q = 0, k = 0, v = 0;
    for (int i = 0; i < width; ++i) {
        q += input[i] * matrix[offset + i];
        k += input[i] * matrix[matrix_elements + offset + i];
        v += input[i] * matrix[2 * matrix_elements + offset + i];
    }
    queries[row * width + col] = q;
    keys[(first_position + row) * width + col] = k;
    values[(first_position + row) * width + col] = v;
}

__global__ void causal_attention(const float* queries, const float* keys, const float* values,
                                 float* output, int first_position) {
    extern __shared__ float scores[];
    __shared__ float partial[attention_threads];
    const int row = blockIdx.x;
    const int start = blockIdx.y * head_width;
    const int lane = threadIdx.x;
    const int length = first_position + row + 1;
    const auto* query = queries + row * width + start;
    float max_score = -CUDART_INF_F;

    for (int previous = lane; previous < length; previous += attention_threads) {
        const auto* key = keys + previous * width + start;
        float sum = 0;
        for (int i = 0; i < head_width; ++i) sum += query[i] * key[i];
        scores[previous] = sum * attention_scale;
        max_score = fmaxf(max_score, scores[previous]);
    }
    partial[lane] = max_score;
    __syncthreads();
    for (int stride = attention_threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        __syncthreads();
    }
    const float largest = partial[0];
    __syncthreads();

    float total = 0;
    for (int previous = lane; previous < length; previous += attention_threads) {
        const float mass = expf(scores[previous] - largest);
        scores[previous] = mass;
        total += mass;
    }
    partial[lane] = total;
    __syncthreads();
    for (int stride = attention_threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }

    if (lane < head_width) {
        float context = 0;
        for (int previous = 0; previous < length; ++previous) {
            context += scores[previous] * values[previous * width + start + lane];
        }
        output[row * width + start + lane] = context / partial[0];
    }
}

__global__ void project_residual(const float* weights, const float* input,
                                 const float* context, float* hidden) {
    const int channel = threadIdx.x;
    const int offset = blockIdx.x * width;
    const auto* matrix_row = weights + channel * width;
    float sum = 0;
    for (int i = 0; i < width; ++i) sum += context[offset + i] * matrix_row[i];
    hidden[offset + channel] = input[offset + channel] + sum;
}

__global__ void choose_token(const float* hidden, unsigned char* output) {
    __shared__ float scores[vocabulary_size];
    const int candidate = threadIdx.x;
    const auto token = vocabulary_byte(candidate);
    float score = 0;
    for (int i = 0; i < width; ++i) score += hidden[i] * feature(token, i, output_salt);
    scores[candidate] = score;
    __syncthreads();
    if (candidate == 0) {
        int best = 0;
        for (int i = 1; i < vocabulary_size; ++i) {
            if (scores[i] > scores[best]) best = i;
        }
        *output = vocabulary_byte(best);
    }
}

class DeviceModel {
public:
    DeviceModel(int capacity, int batch)
        : capacity_(capacity), keys_(layers * capacity * width), values_(layers * capacity * width),
          first_(batch * width), second_(batch * width), queries_(batch * width),
          context_(batch * width), params_(parameters().size()), output_(1) {
        check(cudaMemcpy(params_.get(), parameters().data(),
                         parameters().size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void prefill(std::string_view prompt) {
        const auto count = static_cast<int>(prompt.size());
        DeviceBuffer<unsigned char> tokens(count);
        check(cudaMemcpy(tokens.get(), prompt.data(), prompt.size(), cudaMemcpyHostToDevice));
        embed_prompt<<<count, width>>>(tokens.get(), first_.get());
        check(cudaGetLastError());
        forward(count);
        // Keep prompt bytes valid through the embedding kernel and include completed work in prefill timing.
        check(cudaStreamSynchronize(nullptr));
    }

    void import_cache(const Cache& cache) {
        rows_ = static_cast<int>(cache.size());
        const auto bytes = rows_ * width * sizeof(float);
        for (int layer = 0; layer < layers; ++layer) {
            check(cudaMemcpy(key(layer), cache.attention[layer].keys.data(), bytes, cudaMemcpyHostToDevice));
            check(cudaMemcpy(value(layer), cache.attention[layer].values.data(), bytes, cudaMemcpyHostToDevice));
        }
        hidden_ = first_.get();
        check(cudaMemcpy(hidden_, cache.hidden.data(), cache.hidden.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    Cache export_cache() const {
        Cache cache;
        const auto elements = rows_ * width;
        const auto bytes = elements * sizeof(float);
        for (int layer = 0; layer < layers; ++layer) {
            auto& kv = cache.attention[layer];
            kv.keys.resize(elements);
            kv.values.resize(elements);
            check(cudaMemcpy(kv.keys.data(), key(layer), bytes, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(kv.values.data(), value(layer), bytes, cudaMemcpyDeviceToHost));
        }
        check(cudaMemcpy(cache.hidden.data(), hidden_, cache.hidden.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return cache;
    }

    void generate(int steps, const Emit& emit) {
        for (int i = 0; i < steps; ++i) {
            choose_token<<<1, vocabulary_size>>>(hidden_, output_.get());
            check(cudaGetLastError());
            unsigned char token = 0;
            check(cudaMemcpy(&token, output_.get(), sizeof(token), cudaMemcpyDeviceToHost));
            emit(token);
            if (i + 1 < steps) {
                embed_token<<<1, width>>>(token, rows_, first_.get());
                check(cudaGetLastError());
                forward(1);
            }
        }
    }

private:
    float* key(int layer) const { return keys_.get() + layer * capacity_ * width; }
    float* value(int layer) const { return values_.get() + layer * capacity_ * width; }

    void forward(int count) {
        auto* hidden = first_.get();
        auto* destination = second_.get();
        for (int layer = 0; layer < layers; ++layer) {
            const auto* p = params_.get() + layer * 4 * matrix_elements;
            project<<<count, width>>>(p, hidden, queries_.get(), key(layer), value(layer), rows_);
            check(cudaGetLastError());
            causal_attention<<<dim3(count, heads), attention_threads, (rows_ + count) * sizeof(float)>>>(
                queries_.get(), key(layer), value(layer), context_.get(), rows_);
            check(cudaGetLastError());
            project_residual<<<count, width>>>(p + 3 * matrix_elements, hidden, context_.get(), destination);
            check(cudaGetLastError());
            std::swap(hidden, destination);
        }
        hidden_ = hidden + (count - 1) * width;
        rows_ += count;
    }

    int capacity_;
    int rows_ = 0;
    DeviceBuffer<float> keys_, values_, first_, second_, queries_, context_, params_;
    DeviceBuffer<unsigned char> output_;
    float* hidden_ = nullptr;
};

}  // namespace

Cache prefill_cuda(std::string_view prompt) {
    const auto count = static_cast<int>(prompt.size());
    DeviceModel model(count, count);
    model.prefill(prompt);
    return model.export_cache();
}

void decode_cuda(const Cache& cache, int steps, const Emit& emit) {
    DeviceModel model(static_cast<int>(cache.size()) + steps - 1, 1);
    model.import_cache(cache);
    model.generate(steps, emit);
}

void monolithic_cuda(std::string_view prompt, int steps, const Emit& emit,
                     const std::function<void()>& prefilled) {
    const auto count = static_cast<int>(prompt.size());
    DeviceModel model(count + steps - 1, count);
    model.prefill(prompt);
    prefilled();
    model.generate(steps, emit);
}

}  // namespace fixture
