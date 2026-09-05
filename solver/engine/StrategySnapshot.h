#pragma once

#include "core/Card.h"
#include "game/CompiledGame.h"
#include "game/Identifiers.h"
#include <map>
#include <memory>
#include <vector>

namespace solver::engine
{
struct StrategyEntry
{
    game::InfoSetKey infoSet;
    std::vector<float> probabilities;
};

class StrategySnapshot
{
public:
    StrategySnapshot(std::shared_ptr<const game::CompiledGame> game, std::vector<StrategyEntry> entries);

    const game::CompiledGame& Game() const { return *game_; }
    const std::vector<float>* FindStrategy(const game::InfoSetKey& infoSet) const;
    std::vector<float> StrategyOrUniform(const game::InfoSetKey& infoSet) const;

private:
    using NodeStrategies = std::map<core::HoleCards, std::vector<float>>;

    std::shared_ptr<const game::CompiledGame> game_;
    std::vector<NodeStrategies> strategiesByNode_;
};
} // namespace solver::engine
