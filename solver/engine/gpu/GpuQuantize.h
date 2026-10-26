#pragma once

// 16-bit training state shared by the CPU walk and the CUDA kernels: a hand's
// regrets are int16 and its cumulative strategies uint16, all of a hand's entries at one
// power-of-two scale whose exponent byte follows the node's action rows (see
// HandTraversalData::StateUnits). Ratios within a hand, which regret matching and
// average-strategy normalization use, read the integers directly. An update decodes a
// hand, applies float arithmetic and re-encodes every entry at the scale of the hand's
// largest magnitude, rounding stochastically with a dither derived from the unpadded index
// of the hand's first entry (HandTraversalData::Node::strategyOffset) and the update, so the
// expected stored value is exact and
// increments below one quantum still accumulate. Every device computes the same dither
// and rounding, so equal float inputs quantize identically, and encoding a decoded state
// reproduces its values.
#include "engine/gpu/GpuTypes.h"
#include <cmath>
#include <cstring>

namespace solver::engine::gpu
{
enum : int
{
    // Exponent byte = exponent + bias: zero-initialized state holds the smallest scale
    // 2^-126, and the exponents in use (at most 114) stay well below 255.
    kExponentBias = 126,
    kRegretBits = 15, // magnitudes up to 32767
    kSumBits = 16,    // magnitudes up to 65535
};

GPU_TYPES_INLINE unsigned int FloatBits(float value)
{
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    unsigned int bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

GPU_TYPES_INLINE float BitsFloat(unsigned int bits)
{
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

// 2^exponent for exponents -126 to 127, every one a normal float.
GPU_TYPES_INLINE float PowerOfTwo(int exponent)
{
    return BitsFloat((unsigned int)(exponent + 127) << 23);
}

// The decay of negative regrets over skipped updates: this update's negative discount once per
// skipped update, discount^skipped, which exceeds the product of the skipped updates' own smaller
// discounts, most for long skips early in training. Only longer skips round in the math library.
GPU_TYPES_INLINE float NegativeDecay(unsigned int skipped, float discount)
{
    return skipped > 1 ? exp2f(float(skipped) * log2f(discount)) : skipped ? discount : 1.0f;
}

enum : unsigned int
{
    kDitherSpread = 0x9E3779B9u, // golden-ratio multiplier of the update index
    kDitherMix1 = 0x85EBCA6Bu,   // MurmurHash3's finalizer multipliers
    kDitherMix2 = 0xC2B2AE35u,
};

// A hand's dither in [0, 1) at an update: MurmurHash3's finalizer over the index of
// the hand's first entry and the spread update index.
GPU_TYPES_INLINE float Dither(unsigned int index, unsigned int update)
{
    unsigned int hash = index ^ (update * kDitherSpread);
    hash ^= hash >> 16;
    hash *= kDitherMix1;
    hash ^= hash >> 13;
    hash *= kDitherMix2;
    hash ^= hash >> 16;
    return float(hash >> 8) * (1.0f / 16777216.0f);
}

// The exponent byte of a hand whose largest magnitude must fit bits: the power of two
// leaving that magnitude 2^(bits-1) to 2^bits - 1, one more when it exceeds the limit
// (dithered rounding never carries a magnitude within the limit beyond it), never
// below -126. Zero and denormal magnitudes take the smallest scale.
GPU_TYPES_INLINE unsigned int ExponentByte(float magnitude, int bits)
{
    // With the biased exponent field E, exponent E - 126 - bits leaves the magnitude m * 2^(bits-1)
    // for its mantissa m in [1, 2), above the limit 2^bits - 1 exactly when the mantissa field
    // exceeds 2^23 - 2^(24-bits), which adding 2^(24-bits) - 1 carries into the exponent field.
    // The signed shift sends -0 below the clamp.
    const int carried = int(FloatBits(magnitude) + ((1u << (24 - bits)) - 1u)) >> 23;
    return (unsigned int)((carried < bits ? bits : carried) - bits);
}

GPU_TYPES_INLINE unsigned int RegretExponent(float magnitude)
{
    return ExponentByte(magnitude, kRegretBits);
}

GPU_TYPES_INLINE unsigned int SumExponent(float magnitude)
{
    return ExponentByte(magnitude, kSumBits);
}

// Stochastic floor(value / scale + dither). Truncate first so the signed fractional
// part is exact even near zero; compare it without adding the dither to a large integer.
// Dither's 24-bit grid makes 1 - dither exact, and integral scaled values stay unchanged.
GPU_TYPES_INLINE int Quantize(float value, unsigned int exponentByte, float dither)
{
    const float scaled = value * PowerOfTwo(kExponentBias - int(exponentByte));
#if defined(__CUDA_ARCH__)
    // The comparisons below add the carry of the exact sum scaled + dither to its truncation, so
    // the result is that sum's floor, an integer below 2^16 in magnitude that rounding the sum
    // down keeps.
    return __float2int_rd(__fadd_rd(scaled, dither));
#else
    const int base = int(scaled);
    const float fraction = scaled - float(base);
    return base + (fraction >= 1.0f - dither ? 1 : 0) - (dither < -fraction ? 1 : 0);
#endif
}

// The integer is passed already converted to float, as the kernels hold entries.
GPU_TYPES_INLINE float Dequantize(float quantized, unsigned int exponentByte)
{
    return quantized * PowerOfTwo(int(exponentByte) - kExponentBias);
}

// The mean cumulative sums of hand h's orbit at a decision (HandTraversalData::SymmetrizeSums), from
// the decision's units: its action rows of row units, then its hands' exponent bytes. elements has a
// bit per suit-group element fixing the decision's board, bit 0 the identity, and image(e, k) is the
// index of e^-1 k. The orbit is listed from its smallest member, then that member's images in element
// order, and mean[a] sums its members' sums in that order; both devices use this order and
// arithmetic, so they evaluate equal units. Returns the exponent of the largest mean (SumExponent),
// or kNoIndex for a hand that is its own orbit. The loops stay rolled on the device: unrolled, they
// grow its kernel to tens of thousands of instructions.
template<typename Image>
GPU_TYPES_INLINE unsigned int OrbitMean(
    const unsigned short* units,
    unsigned int actions,
    unsigned int row,
    unsigned int h,
    unsigned int elements,
    const Image& image,
    float* mean
)
{
    unsigned int first = h;
    GPU_ROLLED
    for (unsigned int mask = elements >> 1, e = 1; mask; mask >>= 1, ++e)
        if (mask & 1u)
        {
            const unsigned int member = image(e, h);
            first = member < first ? member : first;
        }
    unsigned int orbit[kMaxSuitGroup] = {first}, size = 1;
    GPU_ROLLED
    for (unsigned int mask = elements >> 1, e = 1; mask; mask >>= 1, ++e)
        if (mask & 1u)
        {
            const unsigned int member = image(e, first);
            bool listed = false;
            GPU_ROLLED
            for (unsigned int k = 0; k < size; ++k)
                listed |= orbit[k] == member;
            if (!listed)
                orbit[size++] = member;
        }
    if (size == 1)
        return kNoIndex;
    const unsigned char* exponents = (const unsigned char*)(units + (unsigned long long)actions * row);
    float magnitude = 0.0f;
    GPU_ROLLED
    for (unsigned int a = 0; a < actions; ++a)
    {
        float sum = 0.0f;
        GPU_ROLLED
        for (unsigned int k = 0; k < size; ++k)
            sum += Dequantize(float(units[(unsigned long long)a * row + orbit[k]]), exponents[orbit[k]]);
        mean[a] = sum / float(size);
        magnitude = mean[a] > magnitude ? mean[a] : magnitude;
    }
    return SumExponent(magnitude);
}
} // namespace solver::engine::gpu
