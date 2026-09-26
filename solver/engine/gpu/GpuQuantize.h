#pragma once

// 16-bit training state shared by the CPU walk and the CUDA kernels: a hand's
// regrets are int16 and its cumulative strategies uint16, all of a hand's entries at one
// power-of-two scale whose exponent byte follows the node's action rows (see
// HandTraversalData::StateUnits). Ratios within a hand, which regret matching and
// average-strategy normalization use, read the integers directly. An update decodes a
// hand, applies float arithmetic and re-encodes every entry at the scale of the hand's
// largest magnitude, rounding stochastically with a dither derived from the index of the
// hand's first entry and the update, so the expected stored value is exact and
// increments below one quantum still accumulate. Every device computes the same dither
// and rounding, so equal float inputs quantize identically, and encoding a decoded state
// reproduces its values.
#include "engine/gpu/GpuTypes.h"
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

// 2^-skipped, the halving of negative regrets over skipped updates, rounded to nearest: a
// normal power of two, a denormal one, or zero below 2^-149.
GPU_TYPES_INLINE float Halving(unsigned int skipped)
{
    return skipped < 127u ? PowerOfTwo(-int(skipped)) : skipped < 150u ? BitsFloat(1u << (149u - skipped)) : 0.0f;
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
} // namespace solver::engine::gpu
