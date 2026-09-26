#include "engine/gpu/GpuPlan.h"
#include "engine/HandEvaluation.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace solver::engine::gpu
{
namespace
{
// Sized so each leaf lane's river regions (one scratch row per slot) mostly stay in the GPU's
// L2 between passes (three lanes on the tuning machine's 64 MB L2); larger batches miss L2,
// smaller ones lose to launch overhead; tests/fixtures/backend-parity.json relies on this
// target splitting its street regions.
constexpr std::size_t kBatchScratchBytes = 12 * 1024 * 1024;

template<typename T>
BufferData Table(const std::vector<T>& values)
{
    return {values.data(), values.size() * sizeof(T)};
}

// The plan's working record of a traversal node: Node's fields unpacked, with the parent's
// index and the first child's index in HandTraversalData::children for walking the tree.
struct LayoutNode
{
    U64 strategy = 0;
    U64 board = 0;
    U32 parent = kNoIndex;
    U32 slot = 0;
    U32 reachSlot[2] = {};
    U32 edge = 0; // first child's index in the edges
    U32 link = 0; // first child's slot; for a showdown with a fold sibling, its parent's slot (see Node)
    U32 count = 0;
    U32 actor = 0;
    NodeKind kind = NodeKind::Decision;
    U32 row = 0;
    U32 rankCounts = 0;
    U32 stamp = 0;
    U32 fold = kNoIndex;
    float foldUtility = 0.0f;
    float utility[3] = {};
};

Node Pack(const LayoutNode& layout)
{
    Node n{};
    n.strategy = layout.strategy;
    n.board = layout.board;
    n.slot = layout.slot;
    std::copy(layout.reachSlot, layout.reachSlot + 2, n.reachSlot);
    n.link = layout.link;
    n.info = PackInfo(layout.kind, layout.actor, layout.parent == kNoIndex, layout.count, layout.row);
    n.rankCounts = layout.rankCounts;
    n.stamp = layout.stamp;
    n.fold = layout.fold;
    n.foldUtility = layout.foldUtility;
    std::copy(layout.utility, layout.utility + 3, n.utility);
    return n;
}

// One rank row's Terminal section for opponent q's hands (see Plan::order), in U32 words: a
// header uint4 (the run groups of warps 2 and 3 in x's two low bytes; y and z, the lane table's
// and the run tokens' offsets in uint4 units), the scan tokens, the lane table and the run
// tokens. Tokens are 16-bit byte offsets of opponent hands in the staged reach, four per uint2;
// unused ones hold the zero slot's, just past the hands. Scan group g of warp segment s
// (LaneSegment) is uint2 g * kWarpSize + s after the header. Warps 2 and 3 have a lane per card
// that some hand of the updating player holds, longest ranked-holder run first, whose table
// entry holds that run's byte offset from the runs' start | its length << 16 | the card << 24
// (0xff << 24 without a card); its group g is run-token uint2 g * 2 * kWarpSize + lane.
struct Section
{
    std::vector<U32> words;
    std::array<U32, 52> runOffset{}, runLength{}; // in floats from the runs' start
};
Section BuildSection(const HandBoardData& tables, std::size_t row, U32 q, U32 hands)
{
    Section section;
    const auto& ranked = tables.rankRows[row][q];
    const U32 zero = U32(sizeof(float)) * hands;
    const U32 rankCount = static_cast<U32>(ranked.hands.size()), span = (rankCount + kWarpSize - 1) / kWarpSize;
    const U32 scanGroups = (span + 3) / 4;
    const auto& mine = tables.holders[1 - q];
    std::vector<U32> lanes;
    for (U32 card = 0; card < 52; ++card)
    {
        section.runLength[card] = ranked.holders.cardOffsets[card + 1] - ranked.holders.cardOffsets[card];
        if (mine.cardOffsets[card + 1] > mine.cardOffsets[card])
            lanes.push_back(card);
    }
    static_assert(52 <= 2 * kWarpSize);
    std::stable_sort(lanes.begin(), lanes.end(), [&](U32 a, U32 b) { return section.runLength[a] > section.runLength[b]; });
    U32 offset = 0, longest[2] = {};
    for (std::size_t k = 0; k < lanes.size(); ++k)
    {
        section.runOffset[lanes[k]] = offset;
        offset += section.runLength[lanes[k]];
        longest[k / kWarpSize] = std::max(longest[k / kWarpSize], section.runLength[lanes[k]]);
    }
    const U32 groups[2] = {(longest[0] + 3) / 4, (longest[1] + 3) / 4};
    const U32 laneTable = 1 + 16 * scanGroups, tokens = laneTable + 16;
    section.words.assign(4 * (tokens + 32 * std::max(groups[0], groups[1])), zero | zero << 16);
    // Token k of a uint2 of words.
    const auto token = [&](std::size_t uint2, U32 k, U32 value)
    {
        U32& word = section.words[2 * uint2 + k / 2];
        word = k % 2 ? (word & 0xffffu) | value << 16 : (word & 0xffff0000u) | value;
    };
    section.words[0] = groups[0] | groups[1] << 8;
    section.words[1] = laneTable;
    section.words[2] = tokens;
    section.words[3] = 0;
    for (U32 segment = 0; segment < kWarpSize; ++segment)
    {
        const auto [begin, end] = LaneSegment(rankCount, segment);
        for (U32 j = 0; j < end - begin; ++j)
            token(2 + (j / 4) * kWarpSize + segment, j % 4, U32(sizeof(float)) * ranked.hands[begin + j]);
    }
    for (U32 k = 0; k < 2 * kWarpSize; ++k)
    {
        U32 lane = 0xffu << 24;
        if (k < lanes.size())
        {
            const U32 card = lanes[k];
            lane = U32(sizeof(float)) * section.runOffset[card] | section.runLength[card] << 16 | card << 24;
            for (U32 j = 0; j < section.runLength[card]; ++j)
                token(
                    2 * std::size_t(tokens) + (j / 4) * 2 * kWarpSize + k,
                    j % 4,
                    U32(sizeof(float)) * ranked.holders.cardLists[ranked.holders.cardOffsets[card] + j]
                );
        }
        section.words[4 * laneTable + k] = lane;
    }
    return section;
}
} // namespace

Plan::Plan(const HandTraversalData& data) : entries(data.strategySize)
{
    if (data.maxActions > kMaxActions)
        throw std::runtime_error(
            "The GPU supports at most " + std::to_string(kMaxActions) + " actions per decision; reduce bet or raise sizes or use CPU"
        );
    const auto& tables = data.tables;
    static_assert(sizeof(Node) == 64 && sizeof(Hand) == 32 && sizeof(State) == 64 && sizeof(Pass) == 40);
    state.board = data.nodes.front().boardMask;
    for (U32 p = 0; p < 2; ++p)
    {
        state.hands[p] = static_cast<U32>(tables.hands[p].size());
        for (const auto& h : tables.hands[p])
            hands.push_back(
                {h.mask,
                 ValueScale(h.opponentMass),
                 U32(h.cardIndices[0]) | U32(h.cardIndices[1]) << 8 | U32(h.matchingOpponent + 1) << 16,
                 h.weight,
                 {}}
            );
    }
    // Rows are padded to whole float4 groups so the kernels move them as 8- and 16-byte vectors.
    state.stride = TerminalLayout::Padded(std::max(state.hands[0], state.hands[1]));
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
    ranks.resize(tables.rankRows.size() * std::size_t(total), 0xffff);
    // Each rank row's section per opponent, then the rows at a pitch that fits the largest.
    std::vector<std::array<Section, 2>> sections(tables.rankRows.size());
    std::size_t sectionWords[2] = {};
    for (std::size_t row = 0; row < tables.rankRows.size(); ++row)
        for (U32 q = 0; q < 2; ++q)
        {
            sections[row][q] = BuildSection(tables, row, q, state.hands[q]);
            sectionWords[q] = std::max(sectionWords[q], sections[row][q].words.size());
        }
    state.orderSection = static_cast<U32>(4 * total + sectionWords[0]);
    state.orderPitch = static_cast<U32>(state.orderSection + sectionWords[1]);
    order.resize(tables.rankRows.size() * state.orderPitch, kNoIndex);
    // Order entries pack two byte offsets from Terminal's staged reach (see TerminalLayout),
    // which fit 16 bits up to the run tables' end.
    static_assert(sizeof(float) * TerminalLayout(HandBoardData::kMaxHands).foldMasses <= 0xffff);
    const auto pack = [](U32 low, U32 high) { return U32(sizeof(float)) * low | U32(sizeof(float)) * high << 16; };
    for (std::size_t row = 0; row < tables.rankRows.size(); ++row)
    {
        const auto rowBase = row * std::size_t(total), orderBase = row * std::size_t(state.orderPitch);
        for (U32 p = 0; p < 2; ++p)
        {
            const auto& source = tables.rankRows[row][p];
            const auto base = p ? state.hands[0] : 0;
            const U32 zero = state.hands[1 - p];
            const TerminalLayout staged(state.hands[1 - p]);
            const Section& section = sections[row][1 - p];
            // The summed reach of a card's first k ranked holders, the zero slot for none.
            const auto run = [&](U32 card, U32 k) { return k ? staged.runs + section.runOffset[card] + k - 1 : zero; };
            for (std::size_t i = 0; i < source.hands.size(); ++i)
            {
                const auto hand = base + source.hands[i];
                ranks[rowBase + hand] = source.ranks[i];
                // A hand's win mass and the ranked mass not stronger than it, which the kernel
                // subtracts from the ranked total for the loss mass; the zero slot stands for none.
                const U32 lower = source.lowerBounds[i], upper = source.upperBounds[i];
                U32 entry[4] = {pack(lower ? staged.forward + lower - 1 : zero, upper ? staged.forward + upper - 1 : zero), 0, 0, 0};
                // Per held card, the blocker positions (at most 51) count its ranked holders below
                // and through the hand's rank, and the run's last entry holds its total.
                U32 totals[2];
                for (U32 c = 0; c < 2; ++c)
                {
                    const auto card = tables.hands[p][source.hands[i]].cardIndices[c];
                    const U32 positions = source.blockers[i] >> (16 * c);
                    entry[1 + c] = pack(run(card, positions & 0xffu), run(card, (positions >> 8) & 0xffu));
                    totals[c] = run(card, section.runLength[card]);
                }
                entry[3] = pack(totals[0], totals[1]);
                std::copy(entry, entry + 4, order.begin() + (orderBase + 4 * hand));
            }
        }
        for (U32 q = 0; q < 2; ++q)
            std::copy(
                sections[row][q].words.begin(),
                sections[row][q].words.end(),
                order.begin() + (orderBase + (q ? state.orderSection : 4 * total))
            );
    }
    // Traversal-ordered layout; the uploaded array repeats each node at its work positions.
    std::vector<LayoutNode> layout(data.nodes.size());
    const auto& edges = data.children;
    for (U32 i = 0; i < layout.size(); ++i)
    {
        const auto& source = data.nodes[i];
        auto& n = layout[i];
        n.strategy = source.strategyOffset;
        n.board = source.boardMask;
        n.edge = static_cast<U32>(source.childOffset);
        n.count = static_cast<U32>(source.childCount);
        n.actor = static_cast<U32>(source.actor);
        n.kind = source.kind;
        n.row = source.kind == NodeKind::ForcedRunout ? static_cast<U32>(HandTraversalData::RunoutRow(source))
                : source.rankRow >= 0                 ? static_cast<U32>(source.rankRow)
                                                      : 0u;
        n.stamp = i;
        if (source.rankRow >= 0)
            for (U32 p = 0; p < 2; ++p)
                n.rankCounts |= static_cast<U32>(tables.rankRows[source.rankRow][p].hands.size()) << (16 * p);
        std::copy(source.utilities.begin(), source.utilities.end(), n.utility);
        if (source.kind == NodeKind::ForcedRunout)
        {
            // Player 0's scales of win and loss counts (see Node).
            const auto scales = HandTraversalData::RunoutScales(source.utilities, n.row);
            n.utility[0] = scales[0];
            n.utility[2] = scales[1];
        }
    }
    for (U32 i = 0; i < layout.size(); ++i)
        for (U32 a = 0; a < layout[i].count; ++a)
            layout[edges[layout[i].edge + a]].parent = i;
    // A fold and a showdown under one decision share the showdown's Terminal block; when
    // they are the decision's only children, that block also backs the decision up, so
    // neither the fold nor the decision is a Terminal or Backup work item. (Propagating the
    // decision's reach there too, instead of in Reach, measured 4% slower.) A showdown's fold
    // holds the fold's traversal index until slots are allocated.
    std::vector<std::uint8_t> fused(layout.size());
    for (U32 i = 0; i < layout.size(); ++i)
    {
        U32 fold = kNoIndex, showdown = kNoIndex;
        for (U32 a = 0; a < layout[i].count; ++a)
        {
            const U32 child = edges[layout[i].edge + a];
            if (layout[child].kind == NodeKind::Fold)
                fold = child;
            else if (layout[child].kind == NodeKind::Showdown)
                showdown = child;
        }
        if (fold == kNoIndex || showdown == kNoIndex)
            continue;
        layout[showdown].fold = fold;
        layout[showdown].foldUtility = layout[fold].utility[0];
        fused[fold] = 1;
        if (layout[i].count == 2)
            fused[i] = 1;
    }
    state.stampCount = static_cast<U32>(layout.size());
    state.outcomeRows = static_cast<U32>(data.runoutRows);
    // Both hand-major layouts of every runout row, so either player's Terminal loop reads
    // its own hands contiguously.
    const auto outcomePairs = std::size_t(state.outcomeRows) * state.hands[0] * state.hands[1];
    outcomeEntries = 2 * outcomePairs;
    if (state.outcomeRows)
    {
        initialization.push_back({Kernel::Outcomes, 0, static_cast<U32>(outcomePairs), OutcomeStage::CountRunouts, 1});
        if (state.outcomeRows > 1)
            initialization.push_back({Kernel::Outcomes, 0, state.hands[0] * state.hands[1], OutcomeStage::SumTurns, 1});
    }

    // Each slot holds one row: the opponent's reach entering its node until the updating
    // player's values replace it.
    const auto bytesPerSlot = sizeof(float) * state.stride;
    const auto batchSlots = std::max<std::size_t>(1, kBatchScratchBytes / bytesPerSlot);
    // A leaf batch adds roots while its cost stays within the target, so only a single leaf
    // root (a chance child without chance nodes below) can exceed it: each leaf lane's
    // scratch starts this far past the previous lane's.
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
    using Ranges = std::vector<std::pair<U32, U32>>;
    // Sorted disjoint nonempty intervals covering the given ones.
    const auto merge = [](Ranges ranges)
    {
        std::sort(ranges.begin(), ranges.end());
        Ranges merged;
        for (const auto& [begin, end] : ranges)
            if (!merged.empty() && begin <= merged.back().second)
                merged.back().second = std::max(merged.back().second, end);
            else if (begin < end)
                merged.emplace_back(begin, end);
        return merged;
    };
    const auto coalesce = [&](const std::vector<U32>& slots)
    {
        Ranges ranges;
        for (const U32 slot : slots)
            ranges.emplace_back(slot, slot + 1);
        return merge(std::move(ranges));
    };
    // Per pass, the slots it writes and those it reads or writes (each slot's scratch row and
    // flag), merged, from which cross-stream predecessors are derived once every pass exists.
    std::vector<Ranges> writes, touches;
    // Player p's updates launch items begin[p] to end[p] of a pass's list.
    using Items = std::array<U32, 2>;
    const auto emit = [&](Kernel kernel, const std::vector<U32>& list, U32 lane, Items begin, Items end, Ranges written, Ranges read)
    {
        if (list.empty())
            return;
        if (work.size() + list.size() > std::numeric_limits<U32>::max())
            throw std::runtime_error("GPU schedule exceeds the supported index range");
        // Reach, Terminal and Backup lanes come from LaunchPass.
        passes.push_back(
            {kernel,
             static_cast<U32>(work.size()),
             static_cast<U32>(list.size()),
             OutcomeStage::None,
             1,
             lane,
             {begin[0], begin[1]},
             {end[0], end[1]}}
        );
        read.insert(read.end(), written.begin(), written.end());
        touches.push_back(merge(std::move(read)));
        writes.push_back(merge(std::move(written)));
        work.insert(work.end(), list.begin(), list.end());
    };
    // A region's passes run in one stream: leaf batches (no chance node below) take the leaf
    // lanes in turn with disjoint scratch, everything above them uses the last lane. Batches with
    // chance nodes below alternate between two scratch copies and are emitted
    // software-pipelined, a batch's Reach passes before the previous batch's Backup passes,
    // so the graph overlaps both with the leaf batches (Plan::predecessors). A region's
    // Terminal pass follows its first batch's Reach passes, so batches sharing its stream
    // are not queued behind it.
    constexpr U32 kLeafLanes = kLaneCount - 1, kSpineLane = kLeafLanes;
    struct Region
    {
        std::vector<std::vector<U32>> levels;
        std::vector<U32> boundary, terminals;
        // Slots its passes touch, each list with the roots' slots in the parent region: inner, for
        // Reach and Terminal, is its own slots less the chance children, which their own boundary
        // Reach passes write and only Backup reads; own, for Backup, is every own slot. parents
        // holds the parent region's reach slots the roots' derivation reads.
        Ranges inner, own, parents;
        std::size_t below; // first slot this region's batches may use
        U32 lane;
    };
    layout[0].slot = layout[0].reachSlot[0] = layout[0].reachSlot[1] = 0;
    // Children of region roots whose boundary pass a player skips (see reach).
    std::vector<std::uint8_t> derived(layout.size());
    // Visits a region, allocating its nodes' children from base, and emits its Reach levels;
    // the caller emits its Terminal pass, its batches and, later, its Backup passes.
    const auto reach = [&](const std::vector<U32>& roots, std::size_t base, std::size_t below, U32 lane) -> Region
    {
        Region region;
        region.lane = lane;
        std::size_t next = base;
        const auto visit = [&](const auto& walk, U32 index, std::size_t depth) -> void
        {
            if (region.levels.size() <= depth)
                region.levels.resize(depth + 1);
            region.levels[depth].push_back(index);
            auto& n = layout[index];
            if (n.kind != NodeKind::Decision && n.kind != NodeKind::Chance && !fused[index])
                region.terminals.push_back(index);
            // A node's children take consecutive slots before any is walked; its record links
            // only the first (see Node).
            n.link = static_cast<U32>(next);
            for (U32 a = 0; a < n.count; ++a)
            {
                const U32 child = edges[n.edge + a];
                if (next >= std::numeric_limits<U32>::max())
                    throw std::runtime_error("GPU scratch exceeds the supported index range");
                layout[child].slot = static_cast<U32>(next++);
                // A decision rewrites only its actor's reach; the other player's stays with its ancestor.
                for (U32 p = 0; p < 2; ++p)
                    layout[child].reachSlot[p] = n.kind == NodeKind::Chance || p == n.actor ? layout[child].slot : n.reachSlot[p];
            }
            // Chance children are region roots, which their own regions walk.
            for (U32 a = 0; a < n.count; ++a)
            {
                const U32 child = edges[n.edge + a];
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
        std::vector<U32> rootSlots, parentSlots, boundarySlots;
        for (U32 root : roots)
        {
            // Streets open without a bet to fold to; the dependency intervals rely on it.
            if (fused[root])
                throw std::logic_error("GPU plan fuses a region root");
            rootSlots.push_back(layout[root].slot);
            if (layout[root].parent != kNoIndex)
                for (U32 p = 0; p < 2; ++p)
                    parentSlots.push_back(layout[layout[root].parent].reachSlot[p]);
        }
        for (U32 child : region.boundary)
            boundarySlots.push_back(layout[child].slot);
        region.inner = region.own = coalesce(rootSlots);
        region.own.emplace_back(static_cast<U32>(base), static_cast<U32>(next));
        region.parents = coalesce(parentSlots);
        U32 begin = static_cast<U32>(base);
        for (const auto& [holeBegin, holeEnd] : coalesce(boundarySlots))
        {
            if (holeBegin > begin)
                region.inner.emplace_back(begin, holeBegin);
            begin = holeEnd;
        }
        if (next > begin)
            region.inner.emplace_back(begin, static_cast<U32>(next));
        // Region roots derive the opponent's reach; deeper decisions are grouped by
        // actor so each player's launch covers only the opponent's.
        for (std::size_t depth = 0; depth < region.levels.size(); ++depth)
        {
            std::vector<U32> list;
            for (U32 index : region.levels[depth])
                if (depth == 0 || layout[index].kind == NodeKind::Decision)
                    list.push_back(index);
            const U32 count = static_cast<U32>(list.size());
            Items from{}, to{count, count};
            if (depth > 0)
            {
                // Actor 0's decisions lead, which player 1's updates launch; player 0's updates
                // launch the rest.
                const U32 split = static_cast<U32>(
                    std::stable_partition(list.begin(), list.end(), [&](U32 index) { return layout[index].actor == 0; }) - list.begin()
                );
                from = {split, 0};
                to = {count, split};
            }
            // Player q's updates skip the boundary pass, launching none of its items, when every
            // root is q's decision below a chance node and every child of a root is an opponent
            // decision: the pass would only store the opponent's reach entering the roots, and
            // those children, in the next Reach pass, which also reads region.parents, derive it
            // instead (see Node). No terminal reads such a root's row, as the children rewrite the
            // opponent's reach.
            for (U32 q = 0; depth == 0 && q < 2; ++q)
            {
                const auto skippable = [&](U32 root)
                {
                    const LayoutNode& n = layout[root];
                    for (U32 a = 0; a < n.count; ++a)
                    {
                        const LayoutNode& child = layout[edges[n.edge + a]];
                        if (child.kind != NodeKind::Decision || child.actor == q)
                            return false;
                    }
                    return n.parent != kNoIndex && n.kind == NodeKind::Decision && n.actor == q;
                };
                if (!std::all_of(list.begin(), list.end(), skippable))
                    continue;
                to[q] = 0;
                for (U32 root : list)
                    for (U32 a = 0; a < layout[root].count; ++a)
                        derived[edges[layout[root].edge + a]] = 1;
            }
            emit(Kernel::Reach, list, lane, from, to, region.inner, region.parents);
        }
        return region;
    };
    // Terminals read reach rows up to the region root, a showdown also its fold sibling's, and
    // write values over only their own rows, their fold siblings' and fused parents', so the
    // region's other reach rows, which later regions derive their reach from, stay readable
    // while the pass runs; roots are chance children and therefore decisions, never terminals
    // or fused parents (see reach).
    const auto terminal = [&](const Region& region)
    {
        std::vector<U32> written;
        for (U32 index : region.terminals)
        {
            written.push_back(layout[index].slot);
            if (layout[index].fold != kNoIndex)
            {
                written.push_back(layout[layout[index].fold].slot);
                if (fused[layout[index].parent])
                    written.push_back(layout[layout[index].parent].slot);
            }
        }
        const U32 count = static_cast<U32>(region.terminals.size());
        emit(Kernel::Terminal, region.terminals, region.lane, {}, {count, count}, coalesce(written), region.inner);
    };
    // A decision with at most three children below a decision of the other actor with two or
    // three is inlined into that parent: in the parent's actor's updates, where the child does
    // not act and only sums its live children, the parent's Backup reads the child's children in
    // its place (see BackupNode<2> and <3>), so the child's Backup item runs only in its own
    // actor's updates. Each level lists actor 0's inlined decisions, the items both players run,
    // then actor 1's.
    std::vector<std::uint8_t> inlined(layout.size());
    const auto backup = [&](const Region& region)
    {
        for (auto level = region.levels.rbegin(); level != region.levels.rend(); ++level)
        {
            std::vector<U32> own[2], both;
            for (U32 index : *level)
            {
                const LayoutNode& n = layout[index];
                if ((n.kind != NodeKind::Decision && n.kind != NodeKind::Chance) || fused[index])
                    continue;
                // The parent's record describes the child by its first child slot (see Node).
                inlined[index] = n.kind == NodeKind::Decision && n.count <= 3 && n.parent != kNoIndex &&
                                 layout[n.parent].kind == NodeKind::Decision && layout[n.parent].actor != n.actor &&
                                 layout[n.parent].count >= 2 && layout[n.parent].count <= 3 && n.link < (1u << 30);
                (inlined[index] ? own[n.actor] : both).push_back(index);
            }
            std::vector<U32> list = own[0];
            const U32 bothBegin = static_cast<U32>(list.size());
            list.insert(list.end(), both.begin(), both.end());
            const U32 bothEnd = static_cast<U32>(list.size());
            list.insert(list.end(), own[1].begin(), own[1].end());
            emit(Kernel::Backup, list, region.lane, {0, bothBegin}, {bothEnd, static_cast<U32>(list.size())}, region.own, {});
        }
    };
    // Leaf batches take the leaf lanes in turn across regions, so every lane gets an equal share.
    U32 leafBatches = 0;
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
        if (ranges.empty())
            terminal(region);
        const bool leaves = std::none_of(region.boundary.begin(), region.boundary.end(), [&](U32 root) { return chanceBelow[root]; });
        std::optional<Region> pending;
        for (std::size_t b = 0; b < ranges.size(); ++b)
        {
            const U32 lane = !leaves ? kSpineLane : leafBatches++ % kLeafLanes;
            Region sub = leaves ? reach(roots(b), region.below + lane * laneStride, 0, lane)
                                : reach(roots(b), region.below + (b % 2) * copy, region.below + 2 * copy, lane);
            if (b == 0)
                terminal(region);
            if (leaves)
            {
                terminal(sub);
                backup(sub);
                continue;
            }
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
    // Whether two merged lists share a slot: each interval of the shorter list looks up the
    // first interval of the longer one that ends after its start.
    const auto overlap = [](const Ranges& a, const Ranges& b)
    {
        if (a.empty() || b.empty() || a.front().first >= b.back().second || b.front().first >= a.back().second)
            return false;
        const Ranges& shorter = a.size() < b.size() ? a : b;
        const Ranges& longer = a.size() < b.size() ? b : a;
        for (const auto& [begin, end] : shorter)
        {
            const auto next =
                std::upper_bound(longer.begin(), longer.end(), begin, [](U32 slot, const auto& range) { return slot < range.second; });
            if (next != longer.end() && next->first < end)
                return true;
        }
        return false;
    };
    // A pass depends on every earlier pass in another stream that touches a slot it touches
    // when either writes, but waits only for those it does not already follow through stream
    // order or another predecessor (1% of the latest conflicting passes per stream on the
    // benchmark tree, whose CUDA graphs launch faster and train 1% faster without the implied
    // edges). reached[i][lane] is one past the latest pass of that stream pass i follows,
    // itself in its own stream, so pass j precedes pass i exactly when
    // j < reached[i][passes[j].lane]. Scanning back from pass i, only passes found before a pass
    // can follow it, so each predecessor advances the clock as it is found.
    predecessors.assign(passes.size(), {});
    std::vector<std::array<U32, kLaneCount>> reached(passes.size());
    std::array<std::array<U32, kLaneCount>, kLaneCount> latest{}; // reached of each stream's latest pass
    for (std::size_t i = 0; i < passes.size(); ++i)
    {
        const U32 lane = passes[i].lane;
        auto clock = latest[lane];
        for (std::size_t j = i; j-- > 0;)
            if (j >= clock[passes[j].lane] && (overlap(writes[i], touches[j]) || overlap(touches[i], writes[j])))
            {
                predecessors[i].push_back(static_cast<U32>(j));
                for (std::size_t l = 0; l < kLaneCount; ++l)
                    clock[l] = std::max(clock[l], reached[j][l]);
            }
        clock[lane] = static_cast<U32>(i + 1);
        reached[i] = latest[lane] = clock;
    }
    // A showdown's Terminal block takes what it needs of its parent from its own record; the
    // fold's action follows from the siblings' consecutive slots (see Node).
    for (auto& n : layout)
        if (n.fold != kNoIndex)
        {
            const LayoutNode& parent = layout[n.parent];
            n.fold = layout[n.fold].slot;
            n.strategy = parent.strategy;
            n.link = parent.slot;
            n.count = parent.count;
            n.actor = parent.actor;
            n.stamp = parent.stamp;
        }
    // Kernels index nodes by work position, so each pass's items get records in order.
    nodes.reserve(work.size());
    for (const auto& pass : passes)
        for (U32 i = pass.offset; i < pass.offset + pass.count; ++i)
        {
            LayoutNode n = layout[work[i]];
            if (pass.operation == Kernel::Backup && n.kind == NodeKind::Decision)
            {
                // A decision's Backup record describes its inlined children (see Node).
                U32 descriptors[3] = {kNoIndex, kNoIndex, kNoIndex};
                for (U32 a = 0; a < n.count && a < 3; ++a)
                {
                    const U32 child = edges[n.edge + a];
                    if (inlined[child])
                        descriptors[a] = layout[child].link | layout[child].count << 30;
                }
                n.board = U64(descriptors[0]) | U64(descriptors[1]) << 32;
                n.fold = descriptors[2];
            }
            else if (pass.operation == Kernel::Reach)
            {
                // The Reach records of region roots, children of chance nodes, and of derived
                // children derive the root's reach from its chance parent's row without loading
                // the parent's record (see Node).
                const U32 root = derived[work[i]] ? n.parent : work[i], parent = layout[root].parent;
                if (parent != kNoIndex && layout[parent].kind == NodeKind::Chance)
                {
                    const LayoutNode& chance = layout[parent];
                    std::copy(chance.reachSlot, chance.reachSlot + 2, n.reachSlot);
                    n.board = layout[root].board & ~chance.board;
                    n.foldUtility = 1.0f / float(chance.count - 4);
                }
            }
            nodes.push_back(Pack(n));
        }
}

std::array<BufferData, kBufferCount> Plan::Buffers() const
{
    std::array<BufferData, kBufferCount> buffers{};
    buffers[NodesBuffer] = Table(nodes);
    buffers[HandsBuffer] = Table(hands);
    buffers[RanksBuffer] = Table(ranks);
    buffers[RunoutsBuffer] = Table(runouts);
    buffers[OrderBuffer] = Table(order);
    buffers[CardsBuffer] = Table(cards);
    buffers[OutcomesBuffer] = {nullptr, outcomeEntries * sizeof(U32)};
    buffers[RegretsBuffer] = {nullptr, entries * sizeof(std::uint16_t)};
    buffers[SumsBuffer] = buffers[RegretsBuffer];
    buffers[ScratchBuffer] = {nullptr, slots * state.stride * sizeof(float)};
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
