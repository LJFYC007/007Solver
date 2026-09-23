#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace solver::core
{
enum class Street : std::uint8_t
{
    Flop,
    Turn,
    River,
};

class PlayerId
{
public:
    explicit PlayerId(std::uint8_t value) : value_(value)
    {
        if (value_ > 1)
            throw std::out_of_range("PlayerId must be 0 or 1");
    }

    static PlayerId Player0() { return PlayerId(0); }
    static PlayerId Player1() { return PlayerId(1); }

    std::size_t Index() const { return value_; }
    PlayerId Other() const { return PlayerId(static_cast<std::uint8_t>(1 - value_)); }

    friend bool operator==(PlayerId left, PlayerId right) { return left.value_ == right.value_; }
    friend bool operator!=(PlayerId left, PlayerId right) { return !(left == right); }
    friend bool operator<(PlayerId left, PlayerId right) { return left.value_ < right.value_; }

private:
    std::uint8_t value_;
};

inline int BoardCardCount(Street street)
{
    switch (street)
    {
    case Street::Flop:
        return 3;
    case Street::Turn:
        return 4;
    case Street::River:
        return 5;
    }
    throw std::invalid_argument("Unknown street");
}
} // namespace solver::core
