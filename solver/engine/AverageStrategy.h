#pragma once

#include <cstddef>

namespace solver::engine
{
// Normalize one hand across actions. Strides support both traversal's action-major
// rows and snapshot's hand-major rows without changing accumulation order.
inline void NormalizeAverageStrategy(
    const float* sums,
    std::size_t stride,
    std::size_t actionCount,
    float* output,
    std::size_t outputStride
)
{
    double total = 0.0;
    for (std::size_t action = 0; action < actionCount; ++action)
        total += sums[action * stride];
    for (std::size_t action = 0; action < actionCount; ++action)
        output[action * outputStride] = total > 0.0 ? static_cast<float>(sums[action * stride] / total) : 1.0f / actionCount;
}
} // namespace solver::engine
