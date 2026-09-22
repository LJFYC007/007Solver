#include "engine/HandTraversalData.h"
#include "engine/AverageStrategy.h"
#include "engine/ChanceGroups.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
HandTraversalData::HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining)
    : HandTraversalData(std::make_shared<const HandBoardData>(problem, root), root, prepareTraining)
{}

HandTraversalData::HandTraversalData(std::shared_ptr<const HandBoardData> tables, game::NodeId root, bool prepareTraining)
    : tables(std::move(tables)), game(this->tables->game)
{
    const auto rootNode = game->GetNode(root);
    const auto& rootState = rootNode.State();
    if (rootState.board != this->tables->board)
        throw std::invalid_argument("Hand tables belong to a different public board");
    const auto& hands = this->tables->hands;
    rootHalfPot = rootState.pot.Raw() / (2.0f * core::Chips::kUnitsPerChip);
    nodes.reserve(rootNode.TraversalNodeCount());
    children.reserve(rootNode.TraversalNodeCount() - 1);
    dealtCardMasks.reserve(rootNode.TraversalNodeCount() - 1);
    const auto visit = [&](const auto& self, const game::GameNode& source, std::size_t depth) -> std::uint32_t
    {
        const auto& state = source.State();
        std::uint64_t boardMask = 0;
        for (int card = 0; card < state.board.CardCount(); ++card)
            boardMask |= std::uint64_t{1} << state.board.CardAt(card).Index();
        Node node{source.Id(), source.Kind(), state.playerToAct.Index(), children.size(), 0, strategySize, boardMask, state.board};
        node.forcedRunout = source.IsForcedRunout();
        if (node.kind == game::NodeKind::Decision)
        {
            node.childCount = source.BettingEdgeCount();
            strategySize += node.childCount * hands[node.actor].size();
            for (const auto& hand : hands[node.actor])
                infoSetCount += !(hand.mask & boardMask);
            maxActions = std::max(maxActions, node.childCount);
        }
        else if (node.kind == game::NodeKind::Chance && !node.forcedRunout)
            node.childCount = source.ChanceOutcomeCount();
        else
        {
            game::TerminalSettlement settlement{
                state.pot, {rootState.stacks[0] - state.stacks[0], rootState.stacks[1] - state.stacks[1]}, std::nullopt
            };
            if (!node.forcedRunout && source.Terminal().kind == game::TerminalKind::Fold)
            {
                settlement.winner = source.Terminal().foldedPlayer->Other();
                node.utilities[0] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
            }
            else
            {
                for (std::size_t outcome = 0; outcome < 3; ++outcome)
                {
                    settlement.winner = outcome == 1 ? std::nullopt : std::optional<core::PlayerId>(core::PlayerId(outcome == 0 ? 0 : 1));
                    node.utilities[outcome] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
                }
                if (!node.forcedRunout)
                {
                    const int a = state.board.CardAt(3).Index(), b = state.board.CardAt(4).Index();
                    node.rankRow = this->tables->rowsByRunout[std::max(a, b) * (std::max(a, b) - 1) / 2 + std::min(a, b)];
                }
            }
        }
        const auto index = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(node);
        children.resize(children.size() + node.childCount);
        dealtCardMasks.resize(dealtCardMasks.size() + node.childCount);
        maxDepth = std::max(maxDepth, depth + 1);
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const auto child = source.Child(action);
            const auto childIndex = self(self, child, depth + 1);
            children[node.childOffset + action] = childIndex;
            if (node.kind == game::NodeKind::Chance)
                dealtCardMasks[node.childOffset + action] = std::uint64_t{1} << child.State().board.CardAt(state.board.CardCount()).Index();
        }
        return index;
    };
    visit(visit, rootNode, 0);
    if (prepareTraining)
    {
        PrepareChanceTasks();
        if (std::any_of(nodes.begin(), nodes.end(), [](const Node& node) { return node.forcedRunout && node.board.CardCount() == 3; }))
            PrepareFlopRunout();
    }
}

void HandTraversalData::PrepareChanceTasks()
{
    VisitChanceGroups(
        game->GetNode(nodes.front().id),
        [&](const game::GameNode& node, std::uint32_t index, const std::vector<std::uint32_t>& path)
        {
            const auto group = static_cast<std::uint32_t>(chanceGroups_.size());
            chanceGroups_.push_back({index, path});
            for (std::uint32_t action = 0; action < node.ChanceOutcomeCount(); ++action)
                chanceTasks_.push_back({group, action});
        }
    );
}

void HandTraversalData::PrepareFlopRunout()
{
    const auto& hands = tables->hands;
    flopOutcomes_.resize(hands[0].size() * hands[1].size());
    for (const auto& ranks : tables->rankRows)
        for (std::size_t first = 0; first < ranks[0].hands.size(); ++first)
            for (std::size_t second = 0; second < ranks[1].hands.size(); ++second)
            {
                if (ranks[0].masks[first] & ranks[1].masks[second])
                    continue;
                auto& outcomes = flopOutcomes_[ranks[0].hands[first] * hands[1].size() + ranks[1].hands[second]];
                outcomes.wins += ranks[0].ranks[first] > ranks[1].ranks[second];
                outcomes.losses += ranks[0].ranks[first] < ranks[1].ranks[second];
            }
}

StrategySnapshot HandTraversalData::ExportStrategy(std::vector<float> sums) const
{
    std::vector<StrategySnapshot::NodeBlock> snapshotNodes;
    std::vector<core::HoleCards> snapshotHands;
    snapshotHands.reserve(infoSetCount);
    const auto maxHands = std::max(tables->hands[0].size(), tables->hands[1].size());
    std::vector<float> nodeSums(maxActions * maxHands);
    std::size_t writeOffset = 0;
    // Copy each action-major node before writing its compact hand-major output.
    // Output never extends beyond the original node block, so later inputs survive.
    for (const Node& node : nodes)
    {
        if (node.kind != game::NodeKind::Decision)
            continue;
        const auto& hands = tables->hands[node.actor];
        std::copy_n(sums.data() + node.strategyOffset, node.childCount * hands.size(), nodeSums.data());
        const auto handOffset = snapshotHands.size();
        const auto probabilityOffset = writeOffset;
        for (std::size_t hand = 0; hand < hands.size(); ++hand)
        {
            if (hands[hand].mask & node.boardMask)
                continue;
            snapshotHands.push_back(hands[hand].cards);
            NormalizeAverageStrategy(nodeSums.data() + hand, hands.size(), node.childCount, sums.data() + writeOffset, 1);
            writeOffset += node.childCount;
        }
        if (snapshotHands.size() != handOffset)
            snapshotNodes.push_back({node.id, handOffset, probabilityOffset, snapshotHands.size() - handOffset, node.childCount});
    }
    // Keep the allocation: shrinking capacity would require another large buffer.
    sums.resize(writeOffset);
    return StrategySnapshot(game, std::move(snapshotNodes), std::move(snapshotHands), std::move(sums));
}
} // namespace solver::engine
