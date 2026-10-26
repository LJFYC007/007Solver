#pragma once

#include "game/CompiledGame.h"
#include <vector>

namespace solver::engine
{
// The first non-all-in chance boundary on each betting path, in traversal order, with its path.
template<typename Visitor>
void VisitChanceGroups(const game::GameNode& root, Visitor visitGroup)
{
    std::vector<std::uint32_t> path;
    const auto visit = [&](const auto& self, const game::GameNode& node) -> void
    {
        if (node.IsForcedRunout() || node.Kind() == game::NodeKind::Terminal)
            return;
        if (node.Kind() == game::NodeKind::Chance)
        {
            visitGroup(node, path);
            return;
        }
        for (std::uint32_t action = 0; action < node.BettingEdgeCount(); ++action)
        {
            path.push_back(action);
            self(self, node.Child(action));
            path.pop_back();
        }
    };
    visit(visit, root);
}
} // namespace solver::engine
