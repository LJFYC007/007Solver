#pragma once

#include "core/Range.h"
#include "engine/SolveProblem.h"
#include "game/GameSpec.h"
#include <cstddef>
#include <istream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace solver::io
{
// A node lock as the scenario states it: steps from the root, an action index at a decision or the
// dealt card at a chance node, and each listed hand's action weights.
struct LockSpec
{
    std::vector<std::variant<std::size_t, core::Card>> path;
    std::vector<std::pair<core::HoleCards, std::vector<float>>> hands;
};

struct Scenario
{
    game::GameSpec game;
    core::RangeSet ranges;
    int iterations;
    float accuracyPercent = 0.01f;
    std::vector<LockSpec> locks;
};

Scenario LoadScenario(const std::string& jsonPath);
Scenario ReadScenario(std::istream& input);
// The scenario's compiled game and ranges with its locks resolved: sorted by node, with normalized
// probabilities.
std::shared_ptr<const engine::SolveProblem> MakeProblem(const Scenario& scenario);
} // namespace solver::io
