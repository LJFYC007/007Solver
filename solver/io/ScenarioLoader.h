#pragma once

#include "core/Range.h"
#include "game/GameSpec.h"
#include <istream>
#include <string>

namespace solver::io
{
struct Scenario
{
    game::GameSpec game;
    core::RangeSet ranges;
    int iterations;
    // Zero uses the service's current available-memory budget.
    std::uint64_t memoryBudgetBytes = 0;
};

Scenario LoadScenario(const std::string& jsonPath);
Scenario ReadScenario(std::istream& input);
} // namespace solver::io
