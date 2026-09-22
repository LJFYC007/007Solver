#pragma once

#include "game/CompiledGame.h"
#include <vector>

namespace solver::engine
{
// The first non-all-in chance boundary on each betting path, in traversal order.
// Construction and allocation estimates must agree on paths and preorder indices.
template<typename Visitor>
void VisitChanceGroups(const game::GameNode& root, Visitor visitGroup)
{
    std::vector<std::uint32_t> path;
    const auto visit = [&](const auto& self, const game::GameNode& node, std::uint32_t index) -> void
    {
        if (node.IsForcedRunout() || node.Kind() == game::NodeKind::Terminal)
            return;
        if (node.Kind() == game::NodeKind::Chance)
        {
            visitGroup(node, index, path);
            return;
        }
        auto childIndex = index + 1;
        for (std::uint32_t action = 0; action < node.BettingEdgeCount(); ++action)
        {
            const auto child = node.Child(action);
            path.push_back(action);
            self(self, child, childIndex);
            path.pop_back();
            childIndex += static_cast<std::uint32_t>(child.TraversalNodeCount());
        }
    };
    visit(visit, root, 0);
}
} // namespace solver::engine
