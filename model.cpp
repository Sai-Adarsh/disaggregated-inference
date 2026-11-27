#include "feature.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fixture {

const Parameters& parameters() {
    static const auto values = [] {
        Parameters result{};
        for (int matrix = 0; matrix < layers * 4; ++matrix) {
            const auto salt = 0x942c5a47u + 0x9e3779b9u * static_cast<unsigned>(matrix);
            for (int out = 0; out < width; ++out) {
                for (int in = 0; in < width; ++in) {
                    result[matrix * matrix_elements + out * width + in] =
                        0.125F * feature(static_cast<unsigned char>(out), in, salt);
                }
            }
        }
        return result;
    }();
    return values;
}

namespace {

using Row = std::array<float, width>;

float dot(const float* a, const float* b, int count) {
    float sum = 0;
    for (int i = 0; i < count; ++i) sum += a[i] * b[i];
    return sum;
}

void append(Cache& cache, unsigned char token, std::vector<float>& scores) {
    const auto position = cache.size();
    const auto& weights = parameters();
    Row hidden{};
    for (int channel = 0; channel < width; ++channel) {
        hidden[channel] = embedding(token, static_cast<int>(position), channel);
    }

    for (int layer = 0; layer < layers; ++layer) {
        auto& kv = cache.attention[layer];
        const auto* p = weights.data() + layer * 4 * matrix_elements;
        Row query{};
        for (int channel = 0; channel < width; ++channel) {
            const auto offset = channel * width;
            query[channel] = dot(hidden.data(), p + offset, width);
            kv.keys.push_back(dot(hidden.data(), p + matrix_elements + offset, width));
            kv.values.push_back(dot(hidden.data(), p + 2 * matrix_elements + offset, width));
        }

        Row context{};
        scores.resize(position + 1);
        for (int head = 0; head < heads; ++head) {
            const int start = head * head_width;
            float largest = -std::numeric_limits<float>::infinity();
            for (std::size_t previous = 0; previous <= position; ++previous) {
                const auto* key = kv.keys.data() + previous * width + start;
                scores[previous] = dot(query.data() + start, key, head_width) * attention_scale;
                largest = std::max(largest, scores[previous]);
            }
            float normalizer = 0;
            for (std::size_t previous = 0; previous <= position; ++previous) {
                const float mass = std::exp(scores[previous] - largest);
                normalizer += mass;
                const auto* value = kv.values.data() + previous * width + start;
                for (int i = 0; i < head_width; ++i) context[start + i] += mass * value[i];
            }
            for (int i = 0; i < head_width; ++i) context[start + i] /= normalizer;
        }

        for (int channel = 0; channel < width; ++channel) {
            hidden[channel] += dot(context.data(), p + 3 * matrix_elements + channel * width, width);
        }
    }
    cache.hidden = hidden;
}

unsigned char choose_token(const Row& hidden) {
    float best_score = -std::numeric_limits<float>::infinity();
    unsigned char next = vocabulary_byte(0);
    for (int candidate = 0; candidate < vocabulary_size; ++candidate) {
        const auto token = vocabulary_byte(candidate);
        float score = 0;
        for (int channel = 0; channel < width; ++channel) {
            score += hidden[channel] * feature(token, channel, output_salt);
        }
        if (score > best_score) {
            best_score = score;
            next = token;
        }
    }
    return next;
}

void reserve(Cache& cache, std::size_t rows) {
    for (auto& layer : cache.attention) {
        layer.keys.reserve(rows * width);
        layer.values.reserve(rows * width);
    }
}

}  // namespace

Cache prefill_cpu(std::string_view prompt) {
    Cache cache;
    reserve(cache, prompt.size());
    std::vector<float> scores;
    scores.reserve(prompt.size());
    for (const unsigned char token : prompt) append(cache, token, scores);
    return cache;
}

void decode_cpu(Cache cache, int steps, const Emit& emit) {
    const auto capacity = cache.size() + steps - 1;
    reserve(cache, capacity);
    std::vector<float> scores;
    scores.reserve(capacity);
    for (int step = 0; step < steps; ++step) {
        const auto next = choose_token(cache.hidden);
        emit(next);
        if (step + 1 < steps) append(cache, next, scores);
    }
}

}  // namespace fixture
