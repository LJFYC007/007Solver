#include "engine/HandTraversalData.h"
#include "engine/AverageStrategy.h"
#include "engine/ChanceGroups.h"
#include "engine/gpu/GpuQuantize.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace solver::engine
{
namespace
{
using SuitMap = HandTraversalData::SuitMap;

int MapCard(int card, const SuitMap& map)
{
    return card / 4 * 4 + map[card % 4];
}

std::uint64_t MapMask(std::uint64_t mask, const SuitMap& map)
{
    std::uint64_t mapped = 0;
    for (int card = 0; card < 52; ++card)
        if (mask >> card & 1u)
            mapped |= std::uint64_t{1} << MapCard(card, map);
    return mapped;
}

SuitMap Inverse(const SuitMap& map)
{
    SuitMap inverse{};
    for (std::uint8_t suit = 0; suit < 4; ++suit)
        inverse[map[suit]] = suit;
    return inverse;
}

// A bit per element of a suit group that fixes a board, bit 0 the identity.
std::uint32_t Stabilizer(const std::vector<SuitMap>& group, std::uint64_t boardMask)
{
    std::uint32_t elements = 0;
    for (std::size_t e = 0; e < group.size(); ++e)
        if (MapMask(boardMask, group[e]) == boardMask)
            elements |= 1u << e;
    return elements;
}

// The stabilizer of a board before the river where it is nontrivial, otherwise zero: the boards whose
// chance nodes merge outcomes and whose decisions evaluation symmetrizes.
std::uint32_t PreRiverSymmetry(const std::vector<SuitMap>& group, int boardCards, std::uint64_t boardMask)
{
    if (boardCards >= 5)
        return 0;
    const auto elements = Stabilizer(group, boardMask);
    // Bit 0 is the identity.
    return elements > 1u ? elements : 0;
}

// The group index of a composed with b (a after b).
std::size_t Compose(const std::vector<SuitMap>& group, std::size_t a, std::size_t b)
{
    SuitMap composed{};
    for (std::size_t suit = 0; suit < 4; ++suit)
        composed[suit] = group[a][group[b][suit]];
    return static_cast<std::size_t>(std::find(group.begin(), group.end(), composed) - group.begin());
}
} // namespace

std::uint64_t HandTraversalData::BoardMask(const core::Board& board)
{
    std::uint64_t mask = 0;
    for (int card = 0; card < board.CardCount(); ++card)
        mask |= std::uint64_t{1} << board.CardAt(card).Index();
    return mask;
}

std::vector<HandTraversalData::SuitMap> HandTraversalData::RootSuitGroup(const SolveProblem& problem)
{
    if (!problem.locks.empty())
        return {};
    const auto& board = problem.game->Spec().initialBoard;
    const auto boardMask = BoardMask(board);
    std::vector<SuitMap> group;
    SuitMap map{0, 1, 2, 3};
    do
        if (MapMask(boardMask, map) == boardMask)
            group.push_back(map);
    while (std::next_permutation(map.begin(), map.end()));
    if (group.size() < 2)
        return {};
    // HandBoardData's hands: every image must be one of equal weight.
    for (std::uint8_t player = 0; player < 2; ++player)
    {
        const auto& range = problem.ranges.For(core::PlayerId(player)).Entries();
        for (const auto& [cards, weight] : range)
        {
            if (weight <= 0.0f || core::Overlaps(cards, board))
                continue;
            const auto pair = cards.Cards();
            for (const auto& element : group)
            {
                const auto image = range.find(
                    core::HoleCards(core::Card(MapCard(pair[0].Index(), element)), core::Card(MapCard(pair[1].Index(), element)))
                );
                if (image == range.end() || image->second != weight)
                    return {};
            }
        }
    }
    return group;
}

HandTraversalData::ChanceOutcomes HandTraversalData::CanonicalOutcomes(const std::vector<SuitMap>& group, std::uint64_t boardMask)
{
    ChanceOutcomes result;
    const auto fixing = Stabilizer(group, boardMask);
    std::size_t outcome = 0;
    for (int card = 0; card < 52; ++card)
    {
        if (boardMask >> card & 1u)
            continue;
        const auto index = outcome++;
        // The card's orbit and the first element mapping it onto each of its cards; only the orbit's
        // smallest card is listed.
        std::uint64_t seen = 0;
        std::uint8_t mask = 0;
        for (std::size_t e = 0; e < group.size(); ++e)
        {
            const int image = MapCard(card, group[e]);
            if (fixing >> e & 1u && !(seen >> image & 1u))
            {
                seen |= std::uint64_t{1} << image;
                mask |= static_cast<std::uint8_t>(1u << e);
            }
        }
        if (seen & ((std::uint64_t{1} << card) - 1))
            continue;
        result.outcomes[result.count] = static_cast<std::uint8_t>(index);
        result.masks[result.count++] = mask;
    }
    return result;
}

HandTraversalData::SubtreeSize HandTraversalData::MeasureSubtree(const game::GameNode& root, const std::vector<SuitMap>& group)
{
    const auto& board = root.State().board;
    const auto boardMask = BoardMask(board);
    // Leaves lay out as the full tree, and so do boards without pre-river symmetry: no chance node
    // follows a river, and below a turn board, chance nodes deal on that board.
    if (root.Kind() == game::NodeKind::Terminal || root.IsForcedRunout() || !PreRiverSymmetry(group, board.CardCount(), boardMask))
        return {root.TraversalNodeCount(), root.DecisionNodeCounts(), root.ActionEntryCounts()};
    SubtreeSize size;
    const auto add = [&](const game::GameNode& child, std::uint64_t copies)
    {
        const auto subtree = MeasureSubtree(child, group);
        size.nodes += copies * subtree.nodes;
        for (std::size_t player = 0; player < 2; ++player)
        {
            size.decisions[player] += copies * subtree.decisions[player];
            size.actions[player] += copies * subtree.actions[player];
            size.symmetricDecisions[player] += copies * subtree.symmetricDecisions[player];
            size.symmetricActions[player] += copies * subtree.symmetricActions[player];
        }
    };
    if (root.Kind() == game::NodeKind::Decision)
    {
        // A decision SymmetricElements applies to.
        const auto actor = root.State().playerToAct.Index();
        size.decisions[actor] = size.symmetricDecisions[actor] = 1;
        size.actions[actor] = size.symmetricActions[actor] = root.BettingEdgeCount();
        for (std::size_t action = 0; action < root.BettingEdgeCount(); ++action)
            add(root.Child(action), 1);
        return size;
    }
    const auto outcomes = CanonicalOutcomes(group, boardMask);
    // A chance node's outcomes share one betting topology, so river outcomes also share one layout.
    if (board.CardCount() == 4)
        add(root.Child(outcomes.outcomes[0]), outcomes.count);
    else
        for (std::size_t outcome = 0; outcome < outcomes.count; ++outcome)
            add(root.Child(outcomes.outcomes[outcome]), 1);
    return size;
}

std::uint32_t HandTraversalData::SymmetricElements(const Node& node) const
{
    return node.kind == Kind::Decision ? PreRiverSymmetry(suitGroup, node.board.CardCount(), node.boardMask) : 0;
}

void HandTraversalData::SymmetrizeDecision(const Node& node, const std::uint16_t* units, std::uint16_t* target) const
{
    const auto hands = tables.hands[node.actor].size(), row = RowUnits(hands), actions = node.childCount;
    const auto elements = SymmetricElements(node);
    const auto image = [&](unsigned e, unsigned k) { return unsigned(handImages[e][node.actor][k]); };
    auto* exponents = ExponentBytes(target, actions, row);
    std::vector<float> mean(actions);
    for (std::size_t h = 0; h < hands; ++h)
    {
        const auto exponent = gpu::OrbitMean(units, unsigned(actions), unsigned(row), unsigned(h), elements, image, mean.data());
        if (exponent == gpu::kNoIndex)
            continue;
        for (std::size_t a = 0; a < actions; ++a)
            target[a * row + h] = static_cast<std::uint16_t>(gpu::Quantize(mean[a], exponent, 0.5f));
        exponents[h] = static_cast<std::uint8_t>(exponent);
    }
}

std::unique_ptr<std::uint16_t[]> HandTraversalData::SymmetrizeSums(std::uint16_t* sums) const
{
    // Uninitialized: the workers write every unit.
    std::unique_ptr<std::uint16_t[]> saved(new std::uint16_t[savedOffsets.back()]);
#pragma omp parallel for schedule(dynamic, 16)
    for (std::int64_t k = 0; k < static_cast<std::int64_t>(symmetricDecisions.size()); ++k)
    {
        const Node& node = nodes[symmetricDecisions[k]];
        std::uint16_t* stored = saved.get() + savedOffsets[k];
        std::copy_n(sums + node.stateOffset, savedOffsets[k + 1] - savedOffsets[k], stored);
        SymmetrizeDecision(node, stored, sums + node.stateOffset);
    }
    return saved;
}

void HandTraversalData::RestoreSums(std::uint16_t* sums, const std::uint16_t* saved) const
{
#pragma omp parallel for schedule(dynamic, 16)
    for (std::int64_t k = 0; k < static_cast<std::int64_t>(symmetricDecisions.size()); ++k)
        std::copy_n(saved + savedOffsets[k], savedOffsets[k + 1] - savedOffsets[k], sums + nodes[symmetricDecisions[k]].stateOffset);
}

void HandTraversalData::PrepareHandImages()
{
    handImages.resize(suitGroup.size());
    // RootSuitGroup checked that every image of a hand is a hand.
    for (std::size_t p = 0; p < 2; ++p)
    {
        const auto& hands = tables.hands[p];
        std::vector<int> byPair(kMaxHands, -1);
        for (std::size_t h = 0; h < hands.size(); ++h)
            byPair[core::CardPairIndex(core::Card(hands[h].cardIndices[0]), core::Card(hands[h].cardIndices[1]))] = int(h);
        for (std::size_t element = 0; element < suitGroup.size(); ++element)
        {
            const auto inverse = Inverse(suitGroup[element]);
            auto& images = handImages[element][p];
            images.resize(hands.size());
            for (std::size_t h = 0; h < hands.size(); ++h)
                images[h] = static_cast<std::uint16_t>(byPair[core::CardPairIndex(
                    core::Card(MapCard(hands[h].cardIndices[0], inverse)), core::Card(MapCard(hands[h].cardIndices[1], inverse))
                )]);
        }
    }
}

HandTraversalData::HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining, std::vector<SuitMap> group)
    : tables(problem, root), suitGroup(std::move(group))
{
    if (Isomorphic())
        PrepareHandImages();
    const auto rootNode = tables.game->GetNode(root);
    const auto& rootState = rootNode.State();
    const auto& hands = tables.hands;
    rootHalfPot = core::ToChipUnits(rootState.pot) / 2.0f;
    // The subtrees below each betting path's first non-all-in chance node (ChanceGroups.h) are
    // visited in parallel, each into the preorder ranges that the sequential visit of the nodes
    // above them reserves from the subtrees' node, decision and action counts (MeasureSubtree).
    // The unpadded entries (Node::strategyOffset) and state units of an actor's decisions with
    // these actions in total.
    const auto decisionUnits = [&](std::size_t actions, std::size_t decisions, std::size_t actor)
    {
        const auto count = hands[actor].size();
        return std::array<std::size_t, 2>{actions * count + decisions * ((count + 1) / 2), StateUnits(actions, count, decisions)};
    };
    // Where a visit writes its next node, child edge, unpadded entry and state unit, and the
    // extremes it saw.
    struct Cursor
    {
        std::uint32_t node = 0;
        std::size_t child = 0, strategy = 0, state = 0, maxActions = 1, maxDepth = 1, runoutRows = 0;
    };
    // A subtree the visit of the nodes above defers: its cursor, depth and root, then the cursor
    // its reservation ends at.
    struct Start
    {
        Cursor at;
        std::size_t depth;
        game::GameNode root;
        Cursor end;
    };
    const auto offsets = [](const Cursor& cursor) { return std::make_tuple(cursor.node, cursor.child, cursor.strategy, cursor.state); };
    std::vector<Start> starts;
    // Placeholders, each overwritten by the visit of its node.
    nodes.assign(MeasureSubtree(rootNode, suitGroup).nodes, Node{rootNode.Id(), Kind::Fold, 0, 0, 0, 0, 0, rootState.board});
    children.resize(nodes.size() - 1);
    dealtCards.resize(nodes.size() - 1);
    if (Isomorphic())
        orbitMasks.resize(nodes.size() - 1);
    const auto visit = [&](const auto& self, const game::GameNode& source, std::size_t depth, Cursor& at, bool above) -> std::uint32_t
    {
        const auto& state = source.State();
        const auto boardMask = BoardMask(state.board);
        ChanceOutcomes outcomes;
        const auto kind = source.Kind() == game::NodeKind::Decision            ? Kind::Decision
                          : source.IsForcedRunout()                            ? Kind::ForcedRunout
                          : source.Kind() == game::NodeKind::Chance            ? Kind::Chance
                          : source.Terminal().kind == game::TerminalKind::Fold ? Kind::Fold
                                                                               : Kind::Showdown;
        Node node{source.Id(), kind, state.playerToAct.Index(), at.child, 0, at.strategy, boardMask, state.board};
        if (kind == Kind::Decision)
        {
            node.childCount = source.BettingEdgeCount();
            node.stateOffset = at.state;
            const auto units = decisionUnits(node.childCount, 1, node.actor);
            at.strategy += units[0];
            at.state += units[1];
            at.maxActions = std::max(at.maxActions, node.childCount);
        }
        else if (kind == Kind::Chance)
        {
            outcomes = CanonicalOutcomes(suitGroup, boardMask);
            node.childCount = outcomes.count;
        }
        else
        {
            game::TerminalSettlement settlement{
                state.pot,
                {rootState.stacks[0] - state.stacks[0], rootState.stacks[1] - state.stacks[1]},
                std::nullopt,
                game::TerminalRake(tables.game->Spec(), state),
            };
            node.rake = settlement.rake;
            if (kind == Kind::Fold)
            {
                settlement.winner = source.Terminal().foldedPlayer->Other();
                node.utilities[0][0] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
                node.utilities[1][0] = -node.utilities[0][0] - settlement.rake;
            }
            else
            {
                for (std::size_t outcome = 0; outcome < 3; ++outcome)
                {
                    settlement.winner = outcome == 1 ? std::nullopt : std::optional<core::PlayerId>(core::PlayerId(outcome == 0 ? 0 : 1));
                    node.utilities[0][outcome] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
                    node.utilities[1][2 - outcome] = -node.utilities[0][outcome] - settlement.rake;
                }
                if (kind == Kind::Showdown)
                    node.rankRow = tables.RankRow(state.board);
                else
                    at.runoutRows = std::max(at.runoutRows, RunoutRow(node) + 1);
            }
        }
        const auto index = at.node++;
        nodes[index] = node;
        at.child += node.childCount;
        at.maxDepth = std::max(at.maxDepth, depth + 1);
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const auto child = source.Child(kind == Kind::Chance ? outcomes.outcomes[action] : action);
            auto childIndex = at.node;
            if (above && kind == Kind::Chance)
            {
                starts.push_back({at, depth + 1, child});
                const auto size = MeasureSubtree(child, suitGroup);
                at.node += static_cast<std::uint32_t>(size.nodes);
                at.child += size.nodes - 1;
                // The subtree's decisions of each player with their actions.
                for (std::size_t player = 0; player < 2; ++player)
                {
                    const auto units = decisionUnits(size.actions[player], size.decisions[player], player);
                    at.strategy += units[0];
                    at.state += units[1];
                }
                starts.back().end = at;
            }
            else
                childIndex = self(self, child, depth + 1, at, above);
            children[node.childOffset + action] = childIndex;
            if (node.kind == Kind::Chance)
            {
                dealtCards[node.childOffset + action] =
                    static_cast<std::uint8_t>(child.State().board.CardAt(state.board.CardCount()).Index());
                if (Isomorphic())
                    orbitMasks[node.childOffset + action] = outcomes.masks[action];
            }
        }
        return index;
    };
    Cursor top;
    visit(visit, rootNode, 0, top, true);
    std::vector<Cursor> ends(starts.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t subtree = 0; subtree < static_cast<std::int64_t>(starts.size()); ++subtree)
    {
        ends[subtree] = starts[subtree].at;
        visit(visit, starts[subtree].root, starts[subtree].depth, ends[subtree], false);
    }
    // The visits must fill exactly the ranges MeasureSubtree reserved.
    if (top.node != nodes.size() || top.child != children.size())
        throw std::logic_error("Traversal layout does not match its measured size");
    for (std::size_t subtree = 0; subtree < starts.size(); ++subtree)
        if (offsets(ends[subtree]) != offsets(starts[subtree].end))
            throw std::logic_error("Traversal subtree does not match its measured size");
    strategySize = top.strategy;
    stateSize = top.state;
    ends.push_back(top);
    for (const auto& end : ends)
    {
        maxActions = std::max(maxActions, end.maxActions);
        maxDepth = std::max(maxDepth, end.maxDepth);
        runoutRows = std::max(runoutRows, end.runoutRows);
    }
    savedOffsets.push_back(0);
    for (std::uint32_t index = 0; index < nodes.size(); ++index)
        if (SymmetricElements(nodes[index]))
        {
            symmetricDecisions.push_back(index);
            savedOffsets.push_back(savedOffsets.back() + StateUnits(nodes[index].childCount, hands[nodes[index].actor].size()));
        }
    PrepareChanceTasks();
    if (prepareTraining)
        PrepareRunoutOutcomes();
    PrepareLocks(problem.locks);
}

std::shared_ptr<const HandTraversalData> HandTraversalData::TrainingLayout(const SolveProblem& problem, bool prepareTraining)
{
    return std::make_shared<const HandTraversalData>(problem, problem.game->Root(), prepareTraining, RootSuitGroup(problem));
}

void HandTraversalData::PrepareLocks(const std::vector<NodeLock>& locks)
{
    for (const auto& lock : locks)
    {
        // Preorder node IDs increase with the traversal index; locks outside this subtree are absent.
        const auto node = std::lower_bound(nodes.begin(), nodes.end(), lock.node, [](const Node& n, game::NodeId id) { return n.id < id; });
        if (node == nodes.end() || node->id != lock.node)
            continue;
        if (lockOffsets.empty())
            lockOffsets.assign(nodes.size(), gpu::kNoIndex);
        const auto& hands = tables.hands[node->actor];
        const auto offset = lockedPolicies.size(), row = LockRow(hands.size());
        lockOffsets[node - nodes.begin()] = static_cast<std::uint32_t>(offset);
        lockedPolicies.resize(offset + node->childCount * row, -1.0f);
        // Resolved locks name range hands that the locked node's board leaves, which the tables list.
        for (const auto& [cards, probabilities] : lock.hands)
        {
            const auto hand =
                std::lower_bound(hands.begin(), hands.end(), cards, [](const Hand& h, core::HoleCards c) { return h.cards < c; });
            for (std::size_t action = 0; action < node->childCount; ++action)
                lockedPolicies[offset + action * row + static_cast<std::size_t>(hand - hands.begin())] = probabilities[action];
        }
    }
}

void HandTraversalData::PrepareChanceTasks()
{
    VisitChanceGroups(
        tables.game->GetNode(nodes.front().id),
        [&](const game::GameNode&, const std::vector<std::uint32_t>& path)
        {
            // Above the groups, traversal children follow the betting edges in both layouts.
            std::uint32_t index = 0;
            for (const auto action : path)
                index = children[nodes[index].childOffset + action];
            const auto group = static_cast<std::uint32_t>(chanceGroups_.size());
            chanceGroups_.push_back({index, path});
            for (std::uint32_t action = 0; action < nodes[index].childCount; ++action)
                chanceTasks_.push_back({group, action});
        }
    );
}

void HandTraversalData::PrepareRunoutOutcomes()
{
    if (!runoutRows)
        return;
    const auto& hands = tables.hands;
    const auto& masks = tables.handMasks;
    const std::size_t pairs = hands[0].size() * hands[1].size();
    runoutOutcomes_.assign(2 * runoutRows * pairs, 0);
    // Each rank row is one unordered pair of undealt cards: a runout of the flop row and,
    // with either card as the turn, a river of that card's turn row.
    for (int high = 0; high < 52; ++high)
        for (int low = 0; low < high; ++low)
        {
            const int row = tables.rowsByRunout[core::CardPairIndex(core::Card(high), core::Card(low))];
            if (row < 0)
                continue;
            std::uint32_t* targets[3] = {runoutOutcomes_.data()};
            std::size_t targetCount = 1;
            for (const int turn : {high, low})
                if (static_cast<std::size_t>(turn) + 1 < runoutRows)
                    targets[targetCount++] = runoutOutcomes_.data() + (static_cast<std::size_t>(turn) + 1) * pairs;
            const auto& ranks = tables.rankRows[row];
            for (std::size_t first = 0; first < ranks[0].hands.size(); ++first)
                for (std::size_t second = 0; second < ranks[1].hands.size(); ++second)
                {
                    const auto hand0 = ranks[0].hands[first], hand1 = ranks[1].hands[second];
                    if (masks[0][hand0] & masks[1][hand1])
                        continue;
                    const std::uint32_t outcome = (ranks[0].ranks[first] > ranks[1].ranks[second] ? 1u : 0u) |
                                                  (ranks[0].ranks[first] < ranks[1].ranks[second] ? 1u << 16 : 0u);
                    for (std::size_t target = 0; target < targetCount; ++target)
                        targets[target][hand0 * hands[1].size() + hand1] += outcome;
                }
        }
    std::uint32_t* transposed = runoutOutcomes_.data() + runoutRows * pairs;
    for (std::size_t row = 0; row < runoutRows; ++row)
        for (std::size_t hand0 = 0; hand0 < hands[0].size(); ++hand0)
            for (std::size_t hand1 = 0; hand1 < hands[1].size(); ++hand1)
                transposed[row * pairs + hand1 * hands[0].size() + hand0] = runoutOutcomes_[row * pairs + hand0 * hands[1].size() + hand1];
}

StrategySnapshot HandTraversalData::ExportStrategy(std::vector<std::uint16_t> sums) const
{
    SymmetrizeSums(sums.data()); // the saved sums are not needed again
    return ExportStrategy(sums.size(), [&](const std::vector<std::size_t>&, const auto& consume) { consume(sums.data()); });
}

StrategySnapshot HandTraversalData::ExportStrategy(std::size_t chunkUnits, const SumsStream& stream) const
{
    // Blocks follow the full tree's preorder, whose node IDs increase; each owns its probability
    // range, so decisions normalize independently below. Each block is a traversal decision's image
    // under a suit-group element, the decision itself without isomorphism.
    std::vector<StrategySnapshot::NodeBlock> blocks;
    std::vector<std::uint32_t> decisions;
    std::vector<std::uint8_t> elements; // each block's suit-group element, the identity without isomorphism
    std::vector<core::HoleCards> handLists;
    std::unordered_map<std::uint64_t, std::array<std::size_t, 2>> lists; // offset and count by board mask and actor
    std::size_t probabilityCount = 0;
    const auto addBlock = [&](game::NodeId id, std::uint64_t boardMask, std::uint32_t index, std::size_t element)
    {
        const Node& node = nodes[index];
        const auto [list, created] = lists.try_emplace(boardMask << 1 | node.actor);
        if (created)
        {
            list->second[0] = handLists.size();
            for (const auto& hand : tables.hands[node.actor])
                if (!(hand.mask & boardMask))
                    handLists.push_back(hand.cards);
            list->second[1] = handLists.size() - list->second[0];
        }
        const auto [handOffset, handCount] = list->second;
        if (handCount == 0)
            return;
        blocks.push_back({id, handOffset, probabilityCount, handCount, node.childCount});
        decisions.push_back(index);
        elements.push_back(static_cast<std::uint8_t>(element));
        probabilityCount += handCount * node.childCount;
    };
    // Full node `full` is the image of traversal node index under element g.
    const auto visit = [&](const auto& self, const game::GameNode& full, std::uint32_t index, std::size_t g) -> void
    {
        const Node& node = nodes[index];
        if (node.kind == Kind::Decision)
            addBlock(full.Id(), BoardMask(full.State().board), index, g);
        if (node.kind != Kind::Chance)
        {
            for (std::size_t action = 0; action < node.childCount; ++action)
                if (!nodes[children[node.childOffset + action]].IsLeaf())
                    self(self, full.Child(action), children[node.childOffset + action], g);
            return;
        }
        // In the traversal's frame, a full outcome is a canonical card's image under the element by
        // which orbitMasks merges it.
        std::array<std::array<std::uint8_t, 2>, 52> sources;
        for (std::size_t action = 0; action < node.childCount; ++action)
            for (unsigned mask = orbitMasks[node.childOffset + action], e = 0; mask; mask >>= 1, ++e)
                if (mask & 1u)
                    sources[MapCard(dealtCards[node.childOffset + action], suitGroup[e])] = {
                        static_cast<std::uint8_t>(action), static_cast<std::uint8_t>(e)
                    };
        const auto inverse = Inverse(suitGroup[g]);
        for (std::size_t outcome = 0; outcome < full.ChanceOutcomeCount(); ++outcome)
        {
            const auto child = full.Child(outcome);
            const auto source = sources[MapCard(child.State().board.CardAt(node.board.CardCount()).Index(), inverse)];
            self(self, child, children[node.childOffset + source[0]], Compose(suitGroup, g, source[1]));
        }
    };
    if (!Isomorphic())
    {
        for (std::uint32_t index = 0; index < nodes.size(); ++index)
            if (nodes[index].kind == Kind::Decision)
                addBlock(nodes[index].id, nodes[index].boardMask, index, 0);
    }
    else
        visit(visit, tables.game->GetNode(nodes.front().id), 0, 0);
    // The blocks by traversal decision, in traversal order (a full layout's already are), so that
    // ranges of whole decisions stream their sums once.
    std::vector<std::uint32_t> images(blocks.size()), starts(nodes.size() + 1); // block indices; each decision's first
    for (const auto index : decisions)
        ++starts[index + 1];
    for (std::size_t index = 0; index < nodes.size(); ++index)
        starts[index + 1] += starts[index];
    for (std::uint32_t block = 0; block < blocks.size(); ++block)
        images[starts[decisions[block]]++] = block;
    // Each range starts at its first image's decision, which ends[range - 1] holds.
    std::vector<std::size_t> ends, firstImages{0};
    for (std::size_t image = 0, begin = 0; image < images.size(); ++image)
    {
        const auto index = decisions[images[image]];
        const Node& node = nodes[index];
        if (node.stateOffset + StateUnits(node.childCount, tables.hands[node.actor].size()) - begin > chunkUnits &&
            index != decisions[images[firstImages.back()]])
        {
            begin = node.stateOffset;
            ends.push_back(begin);
            firstImages.push_back(image);
        }
    }
    ends.push_back(stateSize);
    firstImages.push_back(images.size());
    std::unique_ptr<float[]> probabilities(new float[probabilityCount]);
    std::size_t range = 0;
    stream(
        ends,
        [&](const std::uint16_t* units)
        {
            const std::size_t begin = range ? ends[range - 1] : 0;
        // Workers write every probability, touching fresh pages in parallel.
#pragma omp parallel for schedule(dynamic, 256)
            for (std::int64_t image = firstImages[range]; image < static_cast<std::int64_t>(firstImages[range + 1]); ++image)
            {
                const auto block = images[image];
                const auto index = decisions[block];
                const Node& node = nodes[index];
                const auto& hands = tables.hands[node.actor];
                const float* lock = Lock(index);
                const std::uint16_t* sums = units + (node.stateOffset - begin);
                // The image's hand h plays as the decision's hand map[h].
                const std::uint16_t* map = Isomorphic() ? handImages[elements[block]][node.actor].data() : nullptr;
                float* output = probabilities.get() + blocks[block].probabilityOffset;
                // Action-major sums normalize into the compact hand-major output; locked hands export their locks.
                for (std::size_t h = 0; h < hands.size(); ++h)
                {
                    const std::size_t hand = map ? map[h] : h;
                    if (hands[hand].mask & node.boardMask)
                        continue;
                    if (lock && lock[hand] >= 0.0f)
                        for (std::size_t action = 0; action < node.childCount; ++action)
                            output[action] = lock[action * LockRow(hands.size()) + hand];
                    else
                        NormalizeAverageStrategy(sums + hand, RowUnits(hands.size()), node.childCount, output, 1);
                    output += node.childCount;
                }
            }
            ++range;
        }
    );
    return StrategySnapshot(tables.game, std::move(blocks), std::move(handLists), std::move(probabilities));
}

namespace
{
// One buffer of the quantized layout as floats in the unpadded layout, its exponent slots zero.
template<typename Unit>
std::vector<float> DecodeUnits(const HandTraversalData& data, const std::vector<Unit>& units)
{
    std::vector<float> values(data.strategySize);
#pragma omp parallel for schedule(dynamic, 256)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(data.nodes.size()); ++index)
    {
        const auto& node = data.nodes[index];
        if (node.kind != HandTraversalData::Kind::Decision)
            continue;
        const auto hands = data.tables.hands[node.actor].size(), row = HandTraversalData::RowUnits(hands);
        const Unit* nodeUnits = units.data() + node.stateOffset;
        const auto* exponents = ExponentBytes(nodeUnits, node.childCount, row);
        for (std::size_t action = 0; action < node.childCount; ++action)
            for (std::size_t hand = 0; hand < hands; ++hand)
                values[node.strategyOffset + action * hands + hand] =
                    gpu::Dequantize(static_cast<float>(nodeUnits[action * row + hand]), exponents[hand]);
    }
    return values;
}
} // namespace

TrainingState HandTraversalData::Decode(const QuantizedState& state) const
{
    return {DecodeUnits(*this, state.regrets), DecodeUnits(*this, state.strategySums), state.stamps};
}
} // namespace solver::engine
