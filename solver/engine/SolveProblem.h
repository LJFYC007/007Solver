#pragma once

#include "core/Range.h"
#include "game/CompiledGame.h"
#include <memory>

namespace solver::engine
{
struct SolveProblem
{
    std::shared_ptr<const game::CompiledGame> game;
    core::RangeSet ranges;
};
} // namespace solver::engine
