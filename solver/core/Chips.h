#pragma once

#include <cstdint>

namespace solver::core
{
class Chips
{
public:
    static constexpr std::int32_t kUnitsPerChip = 10;

    constexpr Chips() = default;

    static constexpr Chips FromRaw(std::int32_t rawUnits) { return Chips(rawUnits); }

    constexpr std::int32_t Raw() const { return rawUnits_; }

    friend constexpr Chips operator+(Chips left, Chips right) { return FromRaw(left.rawUnits_ + right.rawUnits_); }
    friend constexpr Chips operator-(Chips left, Chips right) { return FromRaw(left.rawUnits_ - right.rawUnits_); }

    Chips& operator+=(Chips other)
    {
        rawUnits_ += other.rawUnits_;
        return *this;
    }

    Chips& operator-=(Chips other)
    {
        rawUnits_ -= other.rawUnits_;
        return *this;
    }

    friend constexpr bool operator==(Chips left, Chips right) { return left.rawUnits_ == right.rawUnits_; }
    friend constexpr bool operator!=(Chips left, Chips right) { return !(left == right); }
    friend constexpr bool operator<(Chips left, Chips right) { return left.rawUnits_ < right.rawUnits_; }
    friend constexpr bool operator<=(Chips left, Chips right) { return left.rawUnits_ <= right.rawUnits_; }
    friend constexpr bool operator>(Chips left, Chips right) { return right < left; }
    friend constexpr bool operator>=(Chips left, Chips right) { return right <= left; }

private:
    explicit constexpr Chips(std::int32_t rawUnits) : rawUnits_(rawUnits) {}

    std::int32_t rawUnits_ = 0;
};
} // namespace solver::core
