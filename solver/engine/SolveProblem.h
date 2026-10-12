#pragma once

#include "core/Range.h"
#include "game/CompiledGame.h"
#include <memory>
#include <utility>
#include <vector>

namespace solver::engine
{
// A decision whose listed hands play fixed probabilities, one per action summing to one; its other
// hands and every other decision train (solver/ARCHITECTURE.md "Node locks").
struct NodeLock
{
    game::NodeId node;
    std::vector<std::pair<core::HoleCards, std::vector<float>>> hands;
};

struct SolveProblem
{
    std::shared_ptr<const game::CompiledGame> game;
    core::RangeSet ranges;
    std::vector<NodeLock> locks; // sorted by node
};
} // namespace solver::engine
