#pragma once

#include "core/Card.h"
#include <cstdint>
#include <stdexcept>

namespace solver::game
{
class NodeId
{
public:
    explicit NodeId(std::int32_t value) : value_(value)
    {
        if (value_ < 0)
            throw std::out_of_range("NodeId cannot be negative");
    }

    std::int32_t Value() const { return value_; }

    friend bool operator==(NodeId left, NodeId right) { return left.value_ == right.value_; }
    friend bool operator!=(NodeId left, NodeId right) { return !(left == right); }
    friend bool operator<(NodeId left, NodeId right) { return left.value_ < right.value_; }

private:
    std::int32_t value_;
};

struct InfoSetKey
{
    NodeId node;
    core::HoleCards hand;
};
} // namespace solver::game
