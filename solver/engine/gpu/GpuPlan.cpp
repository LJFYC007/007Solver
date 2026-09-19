#include "engine/gpu/GpuPlan.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace solver::engine::gpu
{
namespace
{
constexpr U32 kNoParent = std::numeric_limits<U32>::max();
constexpr std::size_t kBatchScratchBytes = 128 * 1024 * 1024;
} // namespace

Plan::Plan(const HandTraversalData& data) : entries(data.strategySize)
{
    static_assert(sizeof(Node) == 80 && sizeof(Hand) == 32 && sizeof(State) == 32);
    for (U32 p = 0; p < 2; ++p)
    {
        state.hands[p] = static_cast<U32>(data.hands[p].size());
        for (const auto& h : data.hands[p])
            hands.push_back({h.mask, h.weight, h.opponentMass, h.cardIndices[0], h.cardIndices[1], h.matchingOpponent, 0});
    }
    state.stride = std::max(state.hands[0], state.hands[1]);
    const U32 total = state.hands[0] + state.hands[1];
    runouts.assign(data.rowsByRunout.begin(), data.rowsByRunout.end());
    ranks.resize(data.rankRows.size() * total, 0xffff);
    order.resize(data.rankRows.size() * total, kNoParent);
    for (std::size_t row = 0; row < data.rankRows.size(); ++row)
        for (U32 p = 0; p < 2; ++p)
        {
            const auto& source = data.rankRows[row][p];
            const auto base = row * total + (p ? state.hands[0] : 0);
            for (std::size_t i = 0; i < source.hands.size(); ++i)
            {
                ranks[base + source.hands[i]] = source.ranks[i];
                order[base + i] = source.hands[i];
            }
        }
    cards.resize(106);
    for (U32 p = 0; p < 2; ++p)
    {
        for (U32 card = 0; card < 52; ++card)
        {
            cards[p * 53 + card] = static_cast<U32>(cards.size());
            for (U32 h = 0; h < state.hands[p]; ++h)
                if (data.hands[p][h].mask & (U64{1} << card))
                    cards.push_back(h);
        }
        cards[p * 53 + 52] = static_cast<U32>(cards.size());
    }
    nodes.resize(data.nodes.size());
    edges = data.children;
    for (U32 i = 0; i < nodes.size(); ++i)
    {
        const auto& source = data.nodes[i];
        auto& n = nodes[i];
        n.strategy = source.strategyOffset;
        n.board = source.boardMask;
        n.parent = kNoParent;
        n.edge = static_cast<U32>(source.childOffset);
        n.count = static_cast<U32>(source.childCount);
        n.actor = static_cast<U32>(source.actor);
        n.kind = source.forcedRunout                       ? 4
                 : source.kind == game::NodeKind::Decision ? 0
                 : source.kind == game::NodeKind::Chance   ? 1
                 : source.rankRow < 0                      ? 2
                                                           : 3;
        n.rankRow = static_cast<U32>(source.rankRow);
        if (source.rankRow >= 0)
            for (U32 p = 0; p < 2; ++p)
                n.rankCounts[p] = static_cast<U32>(data.rankRows[source.rankRow][p].hands.size());
        std::copy(source.utilities.begin(), source.utilities.end(), n.utility);
        if (source.forcedRunout)
        {
            n.outcomeRow = source.board.CardCount() == 3 ? 0 : source.board.CardAt(3).Index() + 1;
            state.outcomeRows = std::max(state.outcomeRows, n.outcomeRow + 1);
        }
    }
    for (U32 i = 0; i < nodes.size(); ++i)
        for (U32 a = 0; a < nodes[i].count; ++a)
        {
            auto& child = nodes[edges[nodes[i].edge + a]];
            child.parent = i;
            child.action = a;
            if (nodes[i].kind == 1)
                child.dealtCard = data.nodes[edges[nodes[i].edge + a]].board.CardAt(data.nodes[i].board.CardCount()).Index();
        }
    outcomeEntries = std::size_t(state.outcomeRows) * state.hands[0] * state.hands[1];

    // A street region retains only its live ancestors and child-root results.
    // Descendant batches reuse the same slots after their root values are backed up.
    std::vector<std::size_t> streetCost(nodes.size());
    for (std::size_t i = nodes.size(); i-- > 0;)
    {
        streetCost[i] = 1;
        if (nodes[i].kind == 1)
            streetCost[i] += nodes[i].count;
        else
            for (U32 a = 0; a < nodes[i].count; ++a)
                streetCost[i] += streetCost[edges[nodes[i].edge + a]];
    }
    const auto emit = [&](Kernel kernel, const std::vector<U32>& list)
    {
        if (list.empty())
            return;
        if (work.size() + list.size() > std::numeric_limits<U32>::max())
            throw std::runtime_error("GPU schedule exceeds the supported index range");
        passes.push_back({kernel, static_cast<U32>(work.size()), static_cast<U32>(list.size())});
        work.insert(work.end(), list.begin(), list.end());
    };
    const auto bytesPerSlot = sizeof(float) * (total + 3 * state.stride);
    const auto batchSlots = std::max<std::size_t>(1, kBatchScratchBytes / bytesPerSlot);
    nodes[0].slot = 0;
    const auto region = [&](const auto& self, const std::vector<U32>& roots, std::size_t next) -> void
    {
        std::vector<std::vector<U32>> levels;
        std::vector<U32> terminals, boundary;
        const auto visit = [&](const auto& walk, U32 index, std::size_t depth) -> void
        {
            if (levels.size() <= depth)
                levels.resize(depth + 1);
            levels[depth].push_back(index);
            auto& n = nodes[index];
            if (n.kind >= 2)
                terminals.push_back(index);
            for (U32 a = 0; a < n.count; ++a)
            {
                const U32 child = edges[n.edge + a];
                if (next >= std::numeric_limits<U32>::max())
                    throw std::runtime_error("GPU scratch exceeds the supported index range");
                nodes[child].slot = static_cast<U32>(next++);
                if (n.kind == 1)
                    boundary.push_back(child);
                else
                    walk(walk, child, depth + 1);
            }
        };
        for (U32 root : roots)
            visit(visit, root, 0);
        slots = std::max(slots, next);
        for (const auto& level : levels)
            emit(Kernel::Reach, level);
        emit(Kernel::Prefix, terminals);
        emit(Kernel::Terminal, terminals);
        for (std::size_t begin = 0; begin < boundary.size();)
        {
            auto end = begin;
            std::size_t cost = 0;
            do
            {
                cost += streetCost[boundary[end++]];
            } while (end < boundary.size() && cost + streetCost[boundary[end]] <= batchSlots);
            self(self, std::vector<U32>(boundary.begin() + begin, boundary.begin() + end), next);
            begin = end;
        }
        for (auto level = levels.rbegin(); level != levels.rend(); ++level)
        {
            std::vector<U32> decisions;
            for (U32 index : *level)
                if (nodes[index].kind < 2)
                    decisions.push_back(index);
            emit(Kernel::Backup, decisions);
        }
    };
    region(region, std::vector<U32>{0}, 1);
}

std::uint64_t Plan::DeviceBytes() const
{
    return nodes.size() * sizeof(Node) + edges.size() * sizeof(U32) + hands.size() * sizeof(Hand) + ranks.size() * sizeof(unsigned short) +
           (order.size() + cards.size() + work.size()) * sizeof(U32) + runouts.size() * sizeof(int) + outcomeEntries * sizeof(U32) +
           2 * entries * sizeof(float) + slots * (state.hands[0] + state.hands[1] + 3 * state.stride) * sizeof(float) + sizeof(State);
}
} // namespace solver::engine::gpu
