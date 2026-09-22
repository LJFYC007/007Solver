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

template<typename T>
BufferData Table(const std::vector<T>& values)
{
    return {values.data(), values.size() * sizeof(T)};
}
} // namespace

Plan::Plan(const HandTraversalData& data) : entries(data.strategySize)
{
    const auto& tables = *data.tables;
    static_assert(sizeof(Node) == 80 && sizeof(Hand) == 32 && sizeof(State) == 32 && sizeof(Pass) == 16);
    for (U32 p = 0; p < 2; ++p)
    {
        state.hands[p] = static_cast<U32>(tables.hands[p].size());
        for (const auto& h : tables.hands[p])
            hands.push_back({h.mask, h.weight, h.opponentMass, h.cardIndices[0], h.cardIndices[1], h.matchingOpponent, 0});
    }
    state.stride = std::max(state.hands[0], state.hands[1]);
    const U32 total = state.hands[0] + state.hands[1];
    runouts.assign(tables.rowsByRunout.begin(), tables.rowsByRunout.end());
    ranks.resize(tables.rankRows.size() * total, 0xffff);
    order.resize(tables.rankRows.size() * 2 * total, kNoParent);
    for (std::size_t row = 0; row < tables.rankRows.size(); ++row)
        for (U32 p = 0; p < 2; ++p)
        {
            const auto& source = tables.rankRows[row][p];
            const auto base = row * total + (p ? state.hands[0] : 0);
            const auto orderBase = row * 2 * total + (p ? state.hands[0] : 0);
            for (std::size_t i = 0; i < source.hands.size(); ++i)
            {
                ranks[base + source.hands[i]] = source.ranks[i];
                order[orderBase + i] = source.hands[i];
                order[orderBase + total + source.hands[i]] = U32(source.lowerBounds[i]) | (U32(source.upperBounds[i]) << 16);
            }
        }
    cards.resize(106);
    for (U32 p = 0; p < 2; ++p)
    {
        for (U32 card = 0; card < 52; ++card)
        {
            cards[p * 53 + card] = static_cast<U32>(cards.size());
            for (U32 h = 0; h < state.hands[p]; ++h)
                if (tables.hands[p][h].mask & (U64{1} << card))
                    cards.push_back(h);
        }
        cards[p * 53 + 52] = static_cast<U32>(cards.size());
    }
    nodes.resize(data.nodes.size());
    const auto& edges = data.children;
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
        n.kind = source.forcedRunout                       ? NodeKind::ForcedRunout
                 : source.kind == game::NodeKind::Decision ? NodeKind::Decision
                 : source.kind == game::NodeKind::Chance   ? NodeKind::Chance
                 : source.rankRow < 0                      ? NodeKind::Fold
                                                           : NodeKind::Showdown;
        n.rankRow = static_cast<U32>(source.rankRow);
        if (source.rankRow >= 0)
            for (U32 p = 0; p < 2; ++p)
                n.rankCounts[p] = static_cast<U32>(tables.rankRows[source.rankRow][p].hands.size());
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
            if (nodes[i].kind == NodeKind::Chance)
                child.dealtCard = data.nodes[edges[nodes[i].edge + a]].board.CardAt(data.nodes[i].board.CardCount()).Index();
        }
    outcomeEntries = std::size_t(state.outcomeRows) * state.hands[0] * state.hands[1];
    if (state.outcomeRows)
    {
        initialization.push_back({Kernel::Outcomes, 0, static_cast<U32>(outcomeEntries), OutcomeStage::CountRunouts});
        if (state.outcomeRows > 1)
            initialization.push_back({Kernel::Outcomes, 0, state.hands[0] * state.hands[1], OutcomeStage::SumTurns});
    }

    // A street region retains only its live ancestors and child-root results.
    // Descendant batches reuse the same slots after their root values are backed up.
    std::vector<std::size_t> streetCost(nodes.size());
    for (std::size_t i = nodes.size(); i-- > 0;)
    {
        streetCost[i] = 1;
        if (nodes[i].kind == NodeKind::Chance)
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
        passes.push_back({kernel, static_cast<U32>(work.size()), static_cast<U32>(list.size()), OutcomeStage::None});
        work.insert(work.end(), list.begin(), list.end());
    };
    const auto bytesPerSlot = sizeof(float) * (total + state.stride);
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
            if (n.kind != NodeKind::Decision && n.kind != NodeKind::Chance)
                terminals.push_back(index);
            for (U32 a = 0; a < n.count; ++a)
            {
                const U32 child = edges[n.edge + a];
                if (next >= std::numeric_limits<U32>::max())
                    throw std::runtime_error("GPU scratch exceeds the supported index range");
                nodes[child].slot = static_cast<U32>(next++);
                if (n.kind == NodeKind::Chance)
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
                if (nodes[index].kind == NodeKind::Decision || nodes[index].kind == NodeKind::Chance)
                    decisions.push_back(index);
            emit(Kernel::Backup, decisions);
        }
    };
    region(region, std::vector<U32>{0}, 1);
    childSlots.reserve(edges.size());
    for (U32 child : edges)
        childSlots.push_back(nodes[child].slot);
}

std::array<BufferData, kBufferCount> Plan::Buffers() const
{
    std::array<BufferData, kBufferCount> buffers{};
    buffers[NodesBuffer] = Table(nodes);
    buffers[ChildSlotsBuffer] = Table(childSlots);
    buffers[HandsBuffer] = Table(hands);
    buffers[RanksBuffer] = Table(ranks);
    buffers[RunoutsBuffer] = Table(runouts);
    buffers[OrderBuffer] = Table(order);
    buffers[CardsBuffer] = Table(cards);
    buffers[WorkBuffer] = Table(work);
    buffers[OutcomesBuffer] = {nullptr, outcomeEntries * sizeof(U32)};
    buffers[RegretsBuffer] = {nullptr, entries * sizeof(float)};
    buffers[SumsBuffer] = buffers[RegretsBuffer];
    buffers[ScratchBuffer] = {nullptr, slots * (state.hands[0] + state.hands[1]) * sizeof(float)};
    buffers[ValuesBuffer] = {nullptr, slots * state.stride * sizeof(float)};
    buffers[StateBuffer] = {&state, sizeof(State)};
    return buffers;
}

std::uint64_t Plan::DeviceBytes() const
{
    std::uint64_t bytes = 0;
    for (const auto& buffer : Buffers())
        bytes += buffer.AllocationBytes();
    return bytes;
}

std::uint64_t Plan::HostBytes() const
{
    std::uint64_t bytes = (initialization.size() + passes.size()) * sizeof(Pass);
    for (const auto& buffer : Buffers())
        if (buffer.data)
            bytes += buffer.bytes;
    return bytes;
}
} // namespace solver::engine::gpu
