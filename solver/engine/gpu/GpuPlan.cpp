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
    static_assert(sizeof(Node) == 96 && sizeof(Hand) == 32 && sizeof(State) == 40 && sizeof(Pass) == 24);
    state.board = data.nodes.front().boardMask;
    for (U32 p = 0; p < 2; ++p)
    {
        state.hands[p] = static_cast<U32>(tables.hands[p].size());
        for (const auto& h : tables.hands[p])
            hands.push_back({h.mask, h.weight, h.opponentMass, h.cardIndices[0], h.cardIndices[1], h.matchingOpponent, 0});
    }
    state.stride = std::max(state.hands[0], state.hands[1]);
    const U32 total = state.hands[0] + state.hands[1];
    runouts.assign(tables.rowsByRunout.begin(), tables.rowsByRunout.end());
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
    const auto rankPitch = 3 * std::size_t(total);
    ranks.resize(tables.rankRows.size() * rankPitch, 0xffff);
    order.resize(tables.rankRows.size() * rankPitch, kNoParent);
    for (std::size_t row = 0; row < tables.rankRows.size(); ++row)
    {
        const auto rowBase = row * rankPitch;
        for (U32 p = 0; p < 2; ++p)
        {
            const auto& source = tables.rankRows[row][p];
            const auto base = rowBase + (p ? state.hands[0] : 0);
            for (std::size_t i = 0; i < source.hands.size(); ++i)
            {
                ranks[base + source.hands[i]] = source.ranks[i];
                order[base + i] = source.hands[i];
                order[base + total + source.hands[i]] = U32(source.lowerBounds[i]) | (U32(source.upperBounds[i]) << 16);
            }
        }
        // Each card's hands in rank order share the card list layout; board-blocked hands trail.
        for (U32 q = 0; q < 2; ++q)
            for (U32 card = 0; card < 52; ++card)
            {
                const auto begin = cards[q * 53 + card], end = cards[q * 53 + card + 1];
                std::vector<U32> list(cards.begin() + begin, cards.begin() + end);
                const auto rankOf = [&](U32 hand) { return ranks[rowBase + (q ? state.hands[0] : 0) + hand]; };
                std::stable_sort(list.begin(), list.end(), [&](U32 a, U32 b) { return rankOf(a) < rankOf(b); });
                for (std::size_t k = 0; k < list.size(); ++k)
                    ranks[rowBase + total + begin - 106 + k] = static_cast<unsigned short>(list[k]);
            }
        // Per hand and held card: how many of the opponent's hands with that card rank strictly
        // below, then at most, the hand. Fewer than 256 hands hold any card.
        for (U32 p = 0; p < 2; ++p)
            for (U32 h = 0; h < state.hands[p]; ++h)
            {
                const U32 mineBase = p ? state.hands[0] : 0, oppBase = p ? 0 : state.hands[0];
                const auto rank = ranks[rowBase + mineBase + h];
                if (rank == 0xffff)
                    continue;
                U32 packed = 0;
                for (U32 c = 0; c < 2; ++c)
                {
                    const U32 card = c ? hands[mineBase + h].card1 : hands[mineBase + h].card0;
                    U32 below = 0, through = 0;
                    for (auto e = cards[(1 - p) * 53 + card]; e < cards[(1 - p) * 53 + card + 1]; ++e)
                    {
                        const auto other = ranks[rowBase + oppBase + ranks[rowBase + total + e - 106]];
                        below += other < rank;
                        through += other <= rank;
                    }
                    packed |= (below | (through << 8)) << (16 * c);
                }
                order[rowBase + 2 * total + mineBase + h] = packed;
            }
    }
    // Traversal-ordered layout; the uploaded array repeats each node at its work positions.
    std::vector<Node> layout(data.nodes.size());
    const auto& edges = data.children;
    for (U32 i = 0; i < layout.size(); ++i)
    {
        const auto& source = data.nodes[i];
        auto& n = layout[i];
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
    for (U32 i = 0; i < layout.size(); ++i)
        for (U32 a = 0; a < layout[i].count; ++a)
        {
            auto& child = layout[edges[layout[i].edge + a]];
            child.parent = i;
            // Terminals never read the updating player's own reach; actions past the first 32 keep the write.
            if (layout[i].kind == NodeKind::Decision && a < 32 && child.kind != NodeKind::Decision && child.kind != NodeKind::Chance)
                layout[i].terminalChildren |= U32{1} << a;
        }
    outcomeEntries = std::size_t(state.outcomeRows) * state.hands[0] * state.hands[1];
    if (state.outcomeRows)
    {
        initialization.push_back({Kernel::Outcomes, 0, static_cast<U32>(outcomeEntries), OutcomeStage::CountRunouts, 1, 0});
        if (state.outcomeRows > 1)
            initialization.push_back({Kernel::Outcomes, 0, state.hands[0] * state.hands[1], OutcomeStage::SumTurns, 1, 0});
    }

    // A street region retains only its live ancestors and child-root results.
    // Descendant batches reuse the same slots after their root values are backed up.
    std::vector<std::size_t> streetCost(layout.size());
    for (std::size_t i = layout.size(); i-- > 0;)
    {
        streetCost[i] = 1;
        if (layout[i].kind == NodeKind::Chance)
            streetCost[i] += layout[i].count;
        else
            for (U32 a = 0; a < layout[i].count; ++a)
                streetCost[i] += streetCost[edges[layout[i].edge + a]];
    }
    std::vector<U32> work;
    const auto emit = [&](Kernel kernel, const std::vector<U32>& list, U32 lanes, bool boundary)
    {
        if (list.empty())
            return;
        if (work.size() + list.size() > std::numeric_limits<U32>::max())
            throw std::runtime_error("GPU schedule exceeds the supported index range");
        passes.push_back(
            {kernel, static_cast<U32>(work.size()), static_cast<U32>(list.size()), OutcomeStage::None, lanes, boundary ? 1u : 0u}
        );
        work.insert(work.end(), list.begin(), list.end());
    };
    const auto bytesPerSlot = sizeof(float) * (total + state.stride);
    const auto batchSlots = std::max<std::size_t>(1, kBatchScratchBytes / bytesPerSlot);
    layout[0].slot = layout[0].reachSlot[0] = layout[0].reachSlot[1] = 0;
    const auto region = [&](const auto& self, const std::vector<U32>& roots, std::size_t next) -> void
    {
        std::vector<std::vector<U32>> levels;
        std::vector<U32> terminals, boundary;
        const auto visit = [&](const auto& walk, U32 index, std::size_t depth) -> void
        {
            if (levels.size() <= depth)
                levels.resize(depth + 1);
            levels[depth].push_back(index);
            auto& n = layout[index];
            if (n.kind != NodeKind::Decision && n.kind != NodeKind::Chance)
                terminals.push_back(index);
            for (U32 a = 0; a < n.count; ++a)
            {
                const U32 child = edges[n.edge + a];
                if (next >= std::numeric_limits<U32>::max())
                    throw std::runtime_error("GPU scratch exceeds the supported index range");
                layout[child].slot = static_cast<U32>(next++);
                // A decision rewrites only its actor's reach; the other player's stays with its ancestor.
                for (U32 p = 0; p < 2; ++p)
                    layout[child].reachSlot[p] = n.kind == NodeKind::Chance || p == n.actor ? layout[child].slot : n.reachSlot[p];
                if (n.kind == NodeKind::Chance)
                    boundary.push_back(child);
                else
                    walk(walk, child, depth + 1);
            }
        };
        for (U32 root : roots)
            visit(visit, root, 0);
        slots = std::max(slots, next);
        // Region roots derive both players' reach; deeper decisions propagate only their actor's.
        for (std::size_t depth = 0; depth < levels.size(); ++depth)
        {
            std::vector<U32> reach;
            U32 lanes = 0;
            for (U32 index : levels[depth])
                if (depth == 0 || layout[index].kind == NodeKind::Decision)
                {
                    reach.push_back(index);
                    lanes = std::max(lanes, depth == 0 ? total : state.hands[layout[index].actor]);
                }
            emit(Kernel::Reach, reach, lanes, depth == 0);
        }
        emit(Kernel::Terminal, terminals, 1, false);
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
                if (layout[index].kind == NodeKind::Decision || layout[index].kind == NodeKind::Chance)
                    decisions.push_back(index);
            emit(Kernel::Backup, decisions, state.stride, false);
        }
    };
    region(region, std::vector<U32>{0}, 1);
    childSlots.reserve(edges.size());
    for (U32 child : edges)
        childSlots.push_back(layout[child].slot);
    for (auto& n : layout)
        for (U32 a = 0; a < n.count && a < 3; ++a)
            n.childSlot[a] = childSlots[n.edge + a];
    // Kernels index nodes by work position, so parents name their Backup entry.
    std::vector<U32> backupPosition(layout.size(), kNoParent);
    for (const auto& pass : passes)
        if (pass.operation == Kernel::Backup)
            for (U32 i = 0; i < pass.count; ++i)
                backupPosition[work[pass.offset + i]] = pass.offset + i;
    nodes.resize(work.size());
    for (std::size_t i = 0; i < work.size(); ++i)
    {
        nodes[i] = layout[work[i]];
        if (nodes[i].parent == kNoParent)
            continue;
        nodes[i].parent = backupPosition[nodes[i].parent];
        if (nodes[i].parent == kNoParent)
            throw std::runtime_error("GPU plan parent lacks a backup entry");
    }
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
