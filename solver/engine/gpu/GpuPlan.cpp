#include "engine/gpu/GpuPlan.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace solver::engine::gpu
{
namespace
{
// Sized so two lanes' river regions keep their reach and values mostly in the GPU's L2
// between passes (64 MB on the tuning machine: 48 and 24 MB were within 2%, 16 MB loses
// to launch overhead); tests/fixtures/backend-parity.json relies on this target splitting
// its street regions.
constexpr std::size_t kBatchScratchBytes = 32 * 1024 * 1024;

template<typename T>
BufferData Table(const std::vector<T>& values)
{
    return {values.data(), values.size() * sizeof(T)};
}
} // namespace

Plan::Plan(const HandTraversalData& data) : entries(data.strategySize)
{
    const auto& tables = *data.tables;
    static_assert(sizeof(Node) == 96 && sizeof(Hand) == 32 && sizeof(State) == 56 && sizeof(Pass) == 36);
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
    cards.resize(kCardListHeader);
    for (U32 p = 0; p < 2; ++p)
    {
        const auto& holders = tables.holders[p];
        for (U32 card = 0; card <= 52; ++card)
            cards[p * kCardListStride + card] = static_cast<U32>(cards.size() + holders.cardOffsets[card]);
        cards.insert(cards.end(), holders.cardLists.begin(), holders.cardLists.end());
    }
    const auto rankPitch = 3 * std::size_t(total);
    ranks.resize(tables.rankRows.size() * rankPitch, 0xffff);
    order.resize(tables.rankRows.size() * rankPitch, kNoIndex);
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
        // Each card's hands in rank order (RankOrder::holders, which the CPU evaluator's
        // blocker positions index) share the card list layout; board-blocked hands trail.
        for (U32 q = 0; q < 2; ++q)
        {
            const auto& ranked = tables.rankRows[row][q].holders;
            const auto& all = tables.holders[q];
            const auto base = rowBase + (q ? state.hands[0] : 0);
            for (U32 card = 0; card < 52; ++card)
            {
                auto list = std::copy(
                    ranked.cardLists.begin() + ranked.cardOffsets[card],
                    ranked.cardLists.begin() + ranked.cardOffsets[card + 1],
                    ranks.begin() + (rowBase + total + cards[q * kCardListStride + card] - kCardListHeader)
                );
                for (auto e = all.cardOffsets[card]; e < all.cardOffsets[card + 1]; ++e)
                    if (ranks[base + all.cardLists[e]] == 0xffff)
                        *list++ = all.cardLists[e];
            }
        }
        for (U32 p = 0; p < 2; ++p)
        {
            const auto& source = tables.rankRows[row][p];
            const U32 base = p ? state.hands[0] : 0;
            for (std::size_t i = 0; i < source.hands.size(); ++i)
                order[rowBase + 2 * total + base + source.hands[i]] = source.blockers[i];
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
        n.parent = kNoIndex;
        n.edge = static_cast<U32>(source.childOffset);
        n.count = static_cast<U32>(source.childCount);
        n.actor = static_cast<U32>(source.actor);
        n.kind = source.kind;
        n.rankRow = static_cast<U32>(source.rankRow);
        n.stamp = i;
        if (source.rankRow >= 0)
            for (U32 p = 0; p < 2; ++p)
                n.rankCounts[p] = static_cast<U32>(tables.rankRows[source.rankRow][p].hands.size());
        std::copy(source.utilities.begin(), source.utilities.end(), n.utility);
        if (source.kind == NodeKind::ForcedRunout)
        {
            n.outcomeRow = source.board.CardCount() == 3 ? 0 : source.board.CardAt(3).Index() + 1;
            state.outcomeRows = std::max(state.outcomeRows, n.outcomeRow + 1);
        }
    }
    for (U32 i = 0; i < layout.size(); ++i)
        for (U32 a = 0; a < layout[i].count; ++a)
            layout[edges[layout[i].edge + a]].parent = i;
    state.stampCount = static_cast<U32>(layout.size());
    outcomeEntries = std::size_t(state.outcomeRows) * state.hands[0] * state.hands[1];
    if (state.outcomeRows)
    {
        initialization.push_back({Kernel::Outcomes, 0, static_cast<U32>(outcomeEntries), OutcomeStage::CountRunouts, 1, 0, 0, 0, 0});
        if (state.outcomeRows > 1)
            initialization.push_back({Kernel::Outcomes, 0, state.hands[0] * state.hands[1], OutcomeStage::SumTurns, 1, 0, 0, 0, 0});
    }

    // A street region retains only its live ancestors and child-root results.
    // Descendant batches reuse the same slots after their root values are backed up.
    std::vector<std::size_t> streetCost(layout.size());
    std::vector<std::uint8_t> chanceBelow(layout.size());
    for (std::size_t i = layout.size(); i-- > 0;)
    {
        streetCost[i] = 1;
        chanceBelow[i] = layout[i].kind == NodeKind::Chance;
        if (layout[i].kind == NodeKind::Chance)
            streetCost[i] += layout[i].count;
        else
            for (U32 a = 0; a < layout[i].count; ++a)
            {
                streetCost[i] += streetCost[edges[layout[i].edge + a]];
                chanceBelow[i] |= chanceBelow[edges[layout[i].edge + a]];
            }
    }
    std::vector<U32> work;
    const auto emit = [&](Kernel kernel, const std::vector<U32>& list, U32 lane, bool boundary, U32 split = 0)
    {
        if (list.empty())
            return;
        if (work.size() + list.size() > std::numeric_limits<U32>::max())
            throw std::runtime_error("GPU schedule exceeds the supported index range");
        // Backup runs one thread per hand of the updating player; Reach lanes come from LaunchPass.
        const U32 lanes = kernel == Kernel::Backup ? state.stride : 1u;
        passes.push_back(
            {kernel,
             static_cast<U32>(work.size()),
             static_cast<U32>(list.size()),
             OutcomeStage::None,
             lanes,
             boundary ? 1u : 0u,
             split,
             lane,
             0}
        );
        work.insert(work.end(), list.begin(), list.end());
    };
    // Each slot holds the opponent's reach and the updating player's values.
    const auto bytesPerSlot = 2 * sizeof(float) * state.stride;
    const auto batchSlots = std::max<std::size_t>(1, kBatchScratchBytes / bytesPerSlot);
    layout[0].slot = layout[0].reachSlot[0] = layout[0].reachSlot[1] = 0;
    const auto region = [&](const auto& self, const std::vector<U32>& roots, std::size_t next, U32 lane) -> void
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
        // Region roots derive the opponent's reach; deeper decisions are grouped by
        // actor so each player's launch covers only the opponent's (see LaunchPass).
        for (std::size_t depth = 0; depth < levels.size(); ++depth)
        {
            std::vector<U32> reach;
            for (U32 index : levels[depth])
                if (depth == 0 || layout[index].kind == NodeKind::Decision)
                    reach.push_back(index);
            U32 split = 0;
            if (depth > 0)
                split = static_cast<U32>(
                    std::stable_partition(reach.begin(), reach.end(), [&](U32 index) { return layout[index].actor == 0; }) - reach.begin()
                );
            emit(Kernel::Reach, reach, lane, depth == 0, split);
        }
        emit(Kernel::Terminal, terminals, lane, false);
        // Batches of leaf regions (no chance node below) alternate between two lanes with
        // disjoint scratch, so one batch's backup can overlap the next batch's reach and
        // terminal passes. Lane 1 forks after this region's last pass and joins before its backup.
        std::vector<std::pair<std::size_t, std::size_t>> batches;
        // Lane 1's scratch starts past the largest lane-0 (even) batch.
        std::size_t laneStride = 0;
        for (std::size_t begin = 0; begin < boundary.size();)
        {
            auto end = begin;
            std::size_t cost = 0;
            do
            {
                cost += streetCost[boundary[end++]];
            } while (end < boundary.size() && cost + streetCost[boundary[end]] <= batchSlots);
            if (batches.size() % 2 == 0)
                laneStride = std::max(laneStride, cost);
            batches.emplace_back(begin, end);
            begin = end;
        }
        // Every region emits a root Reach pass, and a region with batches backs up their chance parents.
        const bool twoLanes =
            lane == 0 && batches.size() > 1 && std::none_of(boundary.begin(), boundary.end(), [&](U32 root) { return chanceBelow[root]; });
        if (twoLanes)
            passes.back().sync |= kForkAfter;
        for (std::size_t b = 0; b < batches.size(); ++b)
        {
            const U32 batchLane = twoLanes && (b & 1) ? 1u : 0u;
            const auto batchPass = passes.size();
            self(
                self,
                std::vector<U32>(boundary.begin() + batches[b].first, boundary.begin() + batches[b].second),
                next + (batchLane ? laneStride : 0),
                batchLane
            );
            if (batchLane)
                passes[batchPass].sync |= kWaitFork;
        }
        const auto joinPass = passes.size();
        for (auto level = levels.rbegin(); level != levels.rend(); ++level)
        {
            std::vector<U32> decisions;
            for (U32 index : *level)
                if (layout[index].kind == NodeKind::Decision || layout[index].kind == NodeKind::Chance)
                    decisions.push_back(index);
            emit(Kernel::Backup, decisions, lane, false);
        }
        if (twoLanes)
            passes[joinPass].sync |= kJoinBefore;
    };
    region(region, std::vector<U32>{0}, 1, 0);
    childSlots.reserve(edges.size());
    for (U32 child : edges)
        childSlots.push_back(layout[child].slot);
    for (auto& n : layout)
        for (U32 a = 0; a < n.count && a < 3; ++a)
            n.childSlot[a] = childSlots[n.edge + a];
    // Kernels index nodes by work position, so parents name their Backup entry.
    std::vector<U32> backupPosition(layout.size(), kNoIndex);
    for (const auto& pass : passes)
        if (pass.operation == Kernel::Backup)
            for (U32 i = 0; i < pass.count; ++i)
                backupPosition[work[pass.offset + i]] = pass.offset + i;
    nodes.resize(work.size());
    for (std::size_t i = 0; i < work.size(); ++i)
    {
        nodes[i] = layout[work[i]];
        if (nodes[i].parent == kNoIndex)
            continue;
        nodes[i].parent = backupPosition[nodes[i].parent];
        if (nodes[i].parent == kNoIndex)
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
    buffers[ScratchBuffer] = {nullptr, slots * state.stride * sizeof(float)};
    buffers[ValuesBuffer] = {nullptr, slots * state.stride * sizeof(float)};
    buffers[FlagsBuffer] = {nullptr, slots * sizeof(U32)};
    buffers[StampsBuffer] = {nullptr, 2 * std::size_t(state.stampCount) * sizeof(U32)};
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
