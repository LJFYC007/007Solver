#pragma once

#include "core/Card.h"
#include "game/CompiledGame.h"
#include "game/Identifiers.h"
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
    // Borrowed probabilities: length is the node's action count; invalidated by destruction, move or assignment.
    const float* FindStrategy(const game::InfoSetKey& infoSet) const;
    std::vector<float> StrategyOrUniform(const game::InfoSetKey& infoSet) const;

private:
    friend class CpuDcfrSession;

    struct NodeBlock
    {
        std::size_t handOffset;
        std::size_t probabilityOffset;
    };

    StrategySnapshot(
        std::shared_ptr<const game::CompiledGame> game,
        const std::vector<game::InfoSetKey>& infoSets,
        std::vector<float> probabilities
    );
    void BuildIndex(const std::vector<game::InfoSetKey>& infoSets);

    std::shared_ptr<const game::CompiledGame> game_;
    std::vector<NodeBlock> nodes_;
    std::vector<core::HoleCards> hands_;
    std::vector<float> probabilities_;
};
} // namespace solver::engine
