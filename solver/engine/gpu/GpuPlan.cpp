#include "engine/gpu/GpuPlan.h"
#include <algorithm>
#include <limits>
#include <optional>
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
    static_assert(sizeof(Node) == 96 && sizeof(Hand) == 32 && sizeof(State) == 56 && sizeof(Pass) == 32);
    state.board = data.nodes.front().boardMask;
    for (U32 p = 0; p < 2; ++p)
    {
        state.hands[p] = static_cast<U32>(tables.hands[p].size());
        // Terminal's per-card run tables lead each card's run with a zero entry, so a run
        // starts at the opponent's card offset plus the card index.
        const auto& holders = tables.holders[1 - p];
        for (const auto& h : tables.hands[p])
        {
            Hand hand{
                h.mask,
                h.opponentMass,
                U32(h.cardIndices[0]) | U32(h.cardIndices[1]) << 8 | U32(h.matchingOpponent + 1) << 16,
                h.weight,
                {},
                0
            };
            for (U32 c = 0; c < 2; ++c)
            {
                const auto card = h.cardIndices[c];
                hand.runs[c] = U32(holders.cardOffsets[card] + card) | U32(holders.cardOffsets[card + 1] - holders.cardOffsets[card]) << 16;
            }
            hands.push_back(hand);
        }
    }
    state.stride = std::max(state.hands[0], state.hands[1]);
    const U32 total = state.hands[0] + state.hands[1];
    state.orderPitch = 3 * total + total % 2;
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
    order.resize(tables.rankRows.size() * state.orderPitch, kNoIndex);
    for (std::size_t row = 0; row < tables.rankRows.size(); ++row)
    {
        const auto rowBase = row * rankPitch, orderBase = row * std::size_t(state.orderPitch);
        for (U32 p = 0; p < 2; ++p)
        {
            const auto& source = tables.rankRows[row][p];
            const auto base = p ? state.hands[0] : 0;
            for (std::size_t i = 0; i < source.hands.size(); ++i)
            {
                const auto hand = base + source.hands[i];
                ranks[rowBase + hand] = source.ranks[i];
                order[orderBase + 2 * hand] = U32(source.lowerBounds[i]) | (U32(source.upperBounds[i]) << 16);
                order[orderBase + 2 * hand + 1] = source.blockers[i];
                order[orderBase + 2 * total + base + i] = source.hands[i];
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
            n.outcomeRow = static_cast<U32>(HandTraversalData::RunoutRow(source));
    }
    for (U32 i = 0; i < layout.size(); ++i)
        for (U32 a = 0; a < layout[i].count; ++a)
            layout[edges[layout[i].edge + a]].parent = i;
    state.stampCount = static_cast<U32>(layout.size());
    state.outcomeRows = static_cast<U32>(data.runoutRows);
    // Both hand-major layouts of every runout row, so either player's Terminal loop reads
    // its own hands contiguously.
    const auto outcomePairs = std::size_t(state.outcomeRows) * state.hands[0] * state.hands[1];
    outcomeEntries = 2 * outcomePairs;
    if (state.outcomeRows)
    {
        initialization.push_back({Kernel::Outcomes, 0, static_cast<U32>(outcomePairs), OutcomeStage::CountRunouts, 1, 0, 0, 0});
        if (state.outcomeRows > 1)
            initialization.push_back({Kernel::Outcomes, 0, state.hands[0] * state.hands[1], OutcomeStage::SumTurns, 1, 0, 0, 0});
    }

    // Each slot holds the opponent's reach and the updating player's values.
    const auto bytesPerSlot = 2 * sizeof(float) * state.stride;
    const auto batchSlots = std::max<std::size_t>(1, kBatchScratchBytes / bytesPerSlot);
    // A leaf batch adds roots while its cost stays within the target, so only a single leaf
    // root (a chance child without chance nodes below) can exceed it: lane 1's scratch
    // starts this far past lane 0's.
    std::size_t laneStride = batchSlots;
    // A street region retains only its live ancestors and child-root results.
    // Descendant batches reuse the same slots after their root values are backed up.
    // A root's cost less one is the slots its region allocates, chance children included.
    std::vector<std::size_t> streetCost(layout.size());
    std::vector<std::uint8_t> chanceBelow(layout.size());
    for (std::size_t i = layout.size(); i-- > 0;)
    {
        streetCost[i] = 1;
        chanceBelow[i] = layout[i].kind == NodeKind::Chance;
        if (layout[i].kind == NodeKind::Chance)
        {
            streetCost[i] += layout[i].count;
            for (U32 a = 0; a < layout[i].count; ++a)
                if (!chanceBelow[edges[layout[i].edge + a]])
                    laneStride = std::max(laneStride, streetCost[edges[layout[i].edge + a]]);
        }
        else
            for (U32 a = 0; a < layout[i].count; ++a)
            {
                streetCost[i] += streetCost[edges[layout[i].edge + a]];
                chanceBelow[i] |= chanceBelow[edges[layout[i].edge + a]];
            }
    }
    std::vector<U32> work;
    // The slot intervals a pass reads or writes, by buffer (reach rows, or values and flags),
    // from which cross-stream predecessors are derived once every pass exists.
    struct Interval
    {
        U32 begin, end;
        bool values, write;
    };
    using Ranges = std::vector<std::pair<U32, U32>>;
    std::vector<std::vector<Interval>> accesses;
    const auto emit = [&](Kernel kernel, const std::vector<U32>& list, U32 lane, bool boundary, U32 split, std::vector<Interval> intervals)
    {
        if (list.empty())
            return;
        if (work.size() + list.size() > std::numeric_limits<U32>::max())
            throw std::runtime_error("GPU schedule exceeds the supported index range");
        // Reach and Backup lanes come from LaunchPass.
        passes.push_back(
            {kernel, static_cast<U32>(work.size()), static_cast<U32>(list.size()), OutcomeStage::None, 1, boundary ? 1u : 0u, split, lane}
        );
        accesses.push_back(std::move(intervals));
        work.insert(work.end(), list.begin(), list.end());
    };
    const auto access = [](std::vector<Interval>& out, const Ranges& ranges, bool values, bool write)
    {
        for (const auto& [begin, end] : ranges)
            out.push_back({begin, end, values, write});
    };
    const auto coalesce = [](std::vector<U32> slots)
    {
        std::sort(slots.begin(), slots.end());
        Ranges ranges;
        for (const U32 slot : slots)
            if (!ranges.empty() && ranges.back().second == slot)
                ++ranges.back().second;
            else if (ranges.empty() || ranges.back().second < slot)
                ranges.emplace_back(slot, slot + 1);
        return ranges;
    };
    // A region's passes run in one stream: leaf batches (no chance node below) alternate
    // lanes 0 and 1 with disjoint scratch, everything above them uses lane 2. Batches with
    // chance nodes below alternate between two scratch copies and are emitted
    // software-pipelined, a batch's Reach and Terminal passes before the previous batch's
    // Backup passes, so the graph overlaps both with the leaf batches (Plan::predecessors).
    constexpr U32 kSpineLane = kLaneCount - 1;
    struct Region
    {
        std::vector<std::vector<U32>> levels;
        std::vector<U32> boundary;
        Ranges own, roots, parents; // own slots; root slots and the reach slots their derivation reads, in the parent region
        std::size_t below;          // first slot this region's batches may use
        U32 lane;
    };
    layout[0].slot = layout[0].reachSlot[0] = layout[0].reachSlot[1] = 0;
    // Visits a region, allocating its nodes' children from base, and emits its Reach levels
    // and Terminal pass; the caller emits its batches and, later, its Backup passes.
    const auto reach = [&](const std::vector<U32>& roots, std::size_t base, std::size_t below, U32 lane) -> Region
    {
        Region region;
        region.lane = lane;
        std::vector<U32> terminals;
        std::size_t next = base;
        const auto visit = [&](const auto& walk, U32 index, std::size_t depth) -> void
        {
            if (region.levels.size() <= depth)
                region.levels.resize(depth + 1);
            region.levels[depth].push_back(index);
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
                    region.boundary.push_back(child);
                else
                    walk(walk, child, depth + 1);
            }
        };
        for (U32 root : roots)
            visit(visit, root, 0);
        slots = std::max(slots, next);
        region.below = below ? below : next;
        region.own = {{static_cast<U32>(base), static_cast<U32>(next)}};
        std::vector<U32> rootSlots, parentSlots;
        for (U32 root : roots)
        {
            rootSlots.push_back(layout[root].slot);
            if (layout[root].parent != kNoIndex)
                for (U32 p = 0; p < 2; ++p)
                    parentSlots.push_back(layout[layout[root].parent].reachSlot[p]);
        }
        region.roots = coalesce(rootSlots);
        region.parents = coalesce(parentSlots);
        // Region roots derive the opponent's reach; deeper decisions are grouped by
        // actor so each player's launch covers only the opponent's (see LaunchPass).
        for (std::size_t depth = 0; depth < region.levels.size(); ++depth)
        {
            std::vector<U32> list;
            for (U32 index : region.levels[depth])
                if (depth == 0 || layout[index].kind == NodeKind::Decision)
                    list.push_back(index);
            U32 split = 0;
            if (depth > 0)
                split = static_cast<U32>(
                    std::stable_partition(list.begin(), list.end(), [&](U32 index) { return layout[index].actor == 0; }) - list.begin()
                );
            std::vector<Interval> intervals;
            access(intervals, region.own, false, true);
            access(intervals, region.roots, false, true);
            access(intervals, region.parents, false, false);
            emit(Kernel::Reach, list, lane, depth == 0, split, std::move(intervals));
        }
        // Terminals read reach rows up to the region root and write their own values;
        // roots are chance children and therefore decisions, never terminals.
        std::vector<Interval> intervals;
        access(intervals, region.own, false, false);
        access(intervals, region.roots, false, false);
        access(intervals, region.own, true, true);
        emit(Kernel::Terminal, terminals, lane, false, 0, std::move(intervals));
        return region;
    };
    const auto backup = [&](const Region& region)
    {
        for (auto level = region.levels.rbegin(); level != region.levels.rend(); ++level)
        {
            std::vector<U32> decisions;
            for (U32 index : *level)
                if (layout[index].kind == NodeKind::Decision || layout[index].kind == NodeKind::Chance)
                    decisions.push_back(index);
            std::vector<Interval> intervals;
            access(intervals, region.own, true, true);
            access(intervals, region.roots, true, true);
            emit(Kernel::Backup, decisions, region.lane, false, 0, std::move(intervals));
        }
    };
    const auto batches = [&](const auto& self, const Region& region) -> void
    {
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        // Slots of the largest batch's regions, one scratch copy of batches with chance nodes below.
        std::size_t copy = 0;
        for (std::size_t begin = 0; begin < region.boundary.size();)
        {
            auto end = begin;
            std::size_t cost = 0;
            do
            {
                cost += streetCost[region.boundary[end++]];
            } while (end < region.boundary.size() && cost + streetCost[region.boundary[end]] <= batchSlots);
            ranges.emplace_back(begin, end);
            copy = std::max(copy, cost - (end - begin));
            begin = end;
        }
        const auto roots = [&](std::size_t b)
        { return std::vector<U32>(region.boundary.begin() + ranges[b].first, region.boundary.begin() + ranges[b].second); };
        if (std::none_of(region.boundary.begin(), region.boundary.end(), [&](U32 root) { return chanceBelow[root]; }))
        {
            for (std::size_t b = 0; b < ranges.size(); ++b)
            {
                const U32 lane = ranges.size() > 1 && (b & 1) ? 1u : 0u;
                const Region leaf = reach(roots(b), region.below + (lane ? laneStride : 0), 0, lane);
                backup(leaf);
            }
            return;
        }
        std::optional<Region> pending;
        for (std::size_t b = 0; b < ranges.size(); ++b)
        {
            Region sub = reach(roots(b), region.below + (b % 2) * copy, region.below + 2 * copy, kSpineLane);
            if (pending)
                backup(*pending);
            self(self, sub);
            pending = std::move(sub);
        }
        if (pending)
            backup(*pending);
    };
    const Region game = reach(std::vector<U32>{0}, 1, 0, kSpineLane);
    batches(batches, game);
    backup(game);
    // A pass depends on every earlier pass in another stream that touches an overlapping
    // slot interval of the same buffer when either writes; stream order covers the rest,
    // so only the latest such pass in each other stream becomes a predecessor.
    predecessors.assign(passes.size(), {});
    std::vector<std::pair<U32, U32>> envelopes;
    for (const auto& intervals : accesses)
    {
        std::pair<U32, U32> envelope{std::numeric_limits<U32>::max(), 0};
        for (const auto& interval : intervals)
            envelope = {std::min(envelope.first, interval.begin), std::max(envelope.second, interval.end)};
        envelopes.push_back(envelope);
    }
    for (std::size_t i = 0; i < passes.size(); ++i)
    {
        std::array<bool, kLaneCount> found{};
        found[passes[i].lane] = true;
        for (std::size_t j = i; j-- > 0;)
        {
            if (found[passes[j].lane] || envelopes[i].first >= envelopes[j].second || envelopes[j].first >= envelopes[i].second)
                continue;
            const bool conflict = std::any_of(
                accesses[i].begin(),
                accesses[i].end(),
                [&](const Interval& a)
                {
                    return std::any_of(
                        accesses[j].begin(),
                        accesses[j].end(),
                        [&](const Interval& b)
                        { return a.values == b.values && (a.write || b.write) && a.begin < b.end && b.begin < a.end; }
                    );
                }
            );
            if (conflict)
            {
                predecessors[i].push_back(static_cast<U32>(j));
                found[passes[j].lane] = true;
            }
        }
    }
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
