#include "model.hpp"
#include "feature.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

namespace fixture {
namespace {

void append(Cache& cache, unsigned char token) {
    for (int i = 0; i < width; ++i) {
        cache.keys.push_back(feature(token, i, key_salt));
        cache.values.push_back(feature(token, i, value_salt));
    }
    cache.last = token;
}

}  // namespace

Cache prefill_cpu(std::string_view prompt) {
    Cache cache;
    cache.keys.reserve(prompt.size() * width);
    cache.values.reserve(prompt.size() * width);
    for (const unsigned char token : prompt) {
        append(cache, token);
    }
    return cache;
}

void decode_cpu(Cache cache, int steps, const Emit& emit) {
    std::vector<float> scores;
    scores.reserve(cache.size() + steps);

    for (int step = 0; step < steps; ++step) {
        std::array<float, width> query{};
        std::array<float, width> context{};
        for (int i = 0; i < width; ++i) {
            query[i] = feature(cache.last, i, query_salt);
        }

        scores.resize(cache.size());
        for (std::size_t row = 0; row < cache.size(); ++row) {
            float score = 0;
            for (int i = 0; i < width; ++i) {
                score += query[i] * cache.keys[row * width + i];
            }
            scores[row] = score * attention_scale;
        }
        const float largest_score = *std::max_element(scores.begin(), scores.end());
        float normalizer = 0;
        for (std::size_t row = 0; row < cache.size(); ++row) {
            const float probability_mass = std::exp(scores[row] - largest_score);
            normalizer += probability_mass;
            for (int i = 0; i < width; ++i) {
                context[i] += probability_mass * cache.values[row * width + i];
            }
        }

        float best_score = -std::numeric_limits<float>::infinity();
        unsigned char next = vocabulary[0];
        for (int candidate = 0; candidate < vocabulary_size; ++candidate) {
            float score = 0;
            for (int i = 0; i < width; ++i) {
                score += context[i] / normalizer * feature(vocabulary[candidate], i, output_salt);
            }
            if (score > best_score) {
                best_score = score;
                next = vocabulary[candidate];
            }
        }
        append(cache, next);
        emit(next);
    }
}

}  // namespace fixture
