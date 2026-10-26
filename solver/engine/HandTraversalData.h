#pragma once

#include "engine/HandBoardData.h"
#include "engine/StrategySnapshot.h"
#include "engine/gpu/GpuTypes.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

namespace solver::engine
{
// Resident DCFR state in the 16-bit layout both backends share (gpu/GpuQuantize.h): per
// decision, action-major rows of the actor's hands, int16 regrets or uint16 cumulative
// strategies, then one exponent byte per hand, every row and the exponent bytes padded to
// whole gpu::StateRow units (see HandTraversalData::StateUnits); and per traversal node the
// update index of the actor's last unpruned update (zero before the first), which discounts
// regrets lazily.
struct QuantizedState
{
    std::vector<std::int16_t> regrets;
    std::vector<std::uint16_t> strategySums;
    std::vector<std::uint32_t> stamps;
};

// The decoded state for lockstep parity checks: one float per entry of the unpadded layout
// (Node::strategyOffset), with the exponent slots zero.
struct TrainingState
{
    std::vector<float> regrets;
    std::vector<float> strategySums;
    std::vector<std::uint32_t> stamps;
};

// The exponent bytes of a decision's hands, following its action rows of row units from units.
template<typename Unit>
inline auto ExponentBytes(Unit* units, std::size_t actions, std::size_t row)
{
    using Byte = std::conditional_t<std::is_const_v<Unit>, const std::uint8_t, std::uint8_t>;
    return reinterpret_cast<Byte*>(units + actions * row);
}

// Immutable, range-specific tables shared by recursive CPU and batched GPU execution.
struct HandTraversalData
{
    static constexpr std::size_t kMaxHands = HandBoardData::kMaxHands;
    using Hand = HandBoardData::Hand;
    using RankOrder = HandBoardData::RankOrder;
    // The GPU layout uses the same kinds. Forced runouts are leaves; only showdowns have a rank row.
    using Kind = gpu::NodeKind;
    struct Node
    {
        game::NodeId id;
        Kind kind;
        std::size_t actor;
        std::size_t childOffset;
        std::size_t childCount;
        // A decision's first entry in the unpadded layout (rows of the actor's hands, then
        // (hands + 1) / 2 exponent units), whose entry indices the dither hashes (see
        // gpu/GpuQuantize.h) and decoded states use.
        std::size_t strategyOffset;
        std::uint64_t boardMask;
        core::Board board;
        int rankRow = -1;
        // Each player's own win/tie/loss payoffs; folds use entry 0.
        std::array<std::array<float, 3>, 2> utilities{};
        float rake = 0.0f;           // what a leaf pays whatever its outcome (game::TerminalRake)
        std::size_t stateOffset = 0; // a decision's first unit of the training state

        bool IsLeaf() const { return kind != Kind::Decision && kind != Kind::Chance; }
    };
    HandBoardData tables;
    std::vector<Node> nodes;
    // 16-bit units of one decision's regrets or strategy sums: its action rows, then its
    // hands' exponent bytes, each padded to gpu::StateRow units so the GPU's hand pairs share
    // words and its warps whole cache lines.
    static std::size_t RowUnits(std::size_t hands) { return gpu::StateRow(static_cast<gpu::U32>(hands)); }
    // Units of decisions with these actions in total.
    static std::size_t StateUnits(std::size_t actions, std::size_t hands, std::size_t decisions = 1)
    {
        return gpu::StateUnits(actions, static_cast<gpu::U32>(hands), decisions);
    }
    std::size_t strategySize = 0; // unpadded entries of the whole tree (Node::strategyOffset)
    std::size_t stateSize = 0;    // units of the whole tree
    std::size_t maxActions = 1;
    float rootHalfPot = 0.0f;

    std::vector<std::uint32_t> children;
    std::vector<std::uint8_t> dealtCards; // card index of each chance edge
    std::size_t maxDepth = 1;

    // CPU walks run one task per listed outcome (canonical cards only in an isomorphic layout) of
    // the first non-all-in chance node on each betting path below the root (the groups of
    // VisitChanceGroups in ChanceGroups.h), each group with its path from the root.
    struct ChanceGroup
    {
        std::uint32_t node;
        std::vector<std::uint32_t> path;
    };
    struct ChanceTask
    {
        std::uint32_t group;
        std::uint32_t action;
    };
    std::vector<ChanceGroup> chanceGroups_;
    std::vector<ChanceTask> chanceTasks_;
    // Runout rows of player-0 win | loss << 16 counts per (player-0 hand, player-1 hand)
    // pair, exact out of the legal runouts and independent of reach/payoffs: row 0 for
    // flop all-ins (990 runouts) and row card + 1 for turn all-ins on that card (44),
    // numbered as the GPU's outcome rows. The player-0-major rows are followed by player-1-major
    // copies so either player's accumulation sweeps its hands contiguously.
    std::vector<std::uint32_t> runoutOutcomes_;
    static std::size_t RunoutRow(const Node& node)
    {
        return node.board.CardCount() == 3 ? 0 : static_cast<std::size_t>(node.board.CardAt(3).Index()) + 1;
    }
    // A chance outcome's probability on a board: every undealt card is an outcome, also where a
    // chance node lists canonical cards only.
    static float OutcomeProbability(const core::Board& board) { return 1.0f / float(52 - board.CardCount() - 4); }
    // The scales of a runout row's win and loss counts under win/tie/loss payoffs: each payoff's
    // difference from the tie per runout. Both devices use these floats.
    static std::array<float, 2> RunoutScales(const std::array<float, 3>& payoffs, std::size_t row)
    {
        const float runouts = row == 0 ? 990.0f : 44.0f;
        return {(payoffs[0] - payoffs[1]) / runouts, (payoffs[2] - payoffs[1]) / runouts};
    }
    std::size_t runoutRows = 0; // one past the largest RunoutRow of a forced runout, zero without any

    // Node locks inside this traversal (solver/ARCHITECTURE.md "Node locks"), empty without any:
    // per node its first entry in lockedPolicies or gpu::kNoIndex. A locked decision's entries are
    // action-major rows of LockRow(actor's hands) floats; a free hand's, and an odd count's padding
    // entry, are -1.
    static std::size_t LockRow(std::size_t hands) { return gpu::LockRow(static_cast<gpu::U32>(hands)); }
    std::vector<std::uint32_t> lockOffsets;
    std::vector<float> lockedPolicies;
    const float* Lock(std::size_t node) const
    {
        return lockOffsets.empty() || lockOffsets[node] == gpu::kNoIndex ? nullptr : lockedPolicies.data() + lockOffsets[node];
    }

    // Suit isomorphism (solver/ARCHITECTURE.md "Suit isomorphism"), all empty without it. suitGroup
    // lists the suit permutations that fix the root board, element 0 the identity, each as the suit
    // it maps each suit to, and handImages[e][p][h] is the index of e^-1 h among player p's hands. A
    // chance node lists only its canonical cards, the smallest card index of each orbit under the
    // elements fixing its board; per chance edge, orbitMasks has a bit per element e that maps the
    // canonical card onto a distinct card of its orbit (bit 0 for the card itself), and the chance
    // node's value for hand h sums v_child(e^-1 h) over those elements.
    using SuitMap = std::array<std::uint8_t, 4>;
    std::vector<SuitMap> suitGroup;
    std::vector<std::array<std::vector<std::uint16_t>, 2>> handImages;
    std::vector<std::uint8_t> orbitMasks;
    bool Isomorphic() const { return !suitGroup.empty(); }
    static std::uint64_t BoardMask(const core::Board& board);
    // The suit group of an isomorphic layout of the game root: the suit permutations that fix its board,
    // the identity first, when there are no locks, the identity is not alone and each element maps each
    // player's hands onto hands of equal weight; otherwise empty.
    static std::vector<SuitMap> RootSuitGroup(const SolveProblem& problem);
    // The outcome indices a chance node on a board lists under a suit group, each orbit's smallest card under
    // the elements fixing the board (every outcome under an empty group), with their orbitMasks.
    struct ChanceOutcomes
    {
        std::size_t count = 0;
        std::array<std::uint8_t, 52> outcomes{}, masks{};
    };
    static ChanceOutcomes CanonicalOutcomes(const std::vector<SuitMap>& group, std::uint64_t boardMask);
    // A subtree's traversal nodes and each player's decisions and their actions in its layout under a suit
    // group, without building it; under an empty group the full layout's (GameNode::TraversalNodeCount).
    // Construction reserves its parallel subtrees by these counts.
    struct SubtreeSize
    {
        std::uint64_t nodes = 1;
        std::array<std::uint64_t, 2> decisions{}, actions{};
        // Of those, the decisions SymmetricElements applies to and their actions.
        std::array<std::uint64_t, 2> symmetricDecisions{}, symmetricActions{};
    };
    static SubtreeSize MeasureSubtree(const game::GameNode& root, const std::vector<SuitMap>& group);
    // Evaluation and export symmetrize the cumulative sums of decisions before the river whose board
    // stabilizer is nontrivial, the decisions that make up reach at merged chance outcomes: a bit per
    // element fixing such a decision's board, zero for other nodes. symmetricDecisions lists those
    // decisions in traversal order, savedOffsets each one's first unit in the saved sums
    // (SymmetrizeSums), then their total.
    std::uint32_t SymmetricElements(const Node& node) const;
    std::vector<std::uint32_t> symmetricDecisions;
    std::vector<std::size_t> savedOffsets;
    // The cumulative sums as evaluation and export read them, as the GPU's gpu::Executor::SymmetrizeSums:
    // SymmetrizeSums writes over the stored sums of symmetricDecisions each hand's mean of its orbit's
    // sums (gpu::OrbitMean), re-encoded at the mean's scale and rounded to nearest, a hand that is its
    // own orbit keeping its own, and returns the stored ones, which RestoreSums puts back.
    std::unique_ptr<std::uint16_t[]> SymmetrizeSums(std::uint16_t* sums) const;
    void RestoreSums(std::uint16_t* sums, const std::uint16_t* saved) const;

    // prepareTraining adds the runout outcome rows, which only CPU training reads. A nonempty group,
    // RootSuitGroup's at the game root only, selects the isomorphic layout.
    HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining = false, std::vector<SuitMap> group = {});
    // The training sessions' layout of the game root, isomorphic under RootSuitGroup.
    static std::shared_ptr<const HandTraversalData> TrainingLayout(const SolveProblem& problem, bool prepareTraining);
    // Normalizes this layout's quantized strategy sums; an actor's decisions on one board share
    // its list of board-compatible hands. An isomorphic layout symmetrizes the sums (SymmetrizeSums)
    // and exports the full tree: a full decision is a traversal decision's image under the
    // composition e of the elements by which orbitMasks merges the chance outcomes above it; its
    // hand h plays as the traversal decision's hand handImages[e][p][h].
    StrategySnapshot ExportStrategy(std::vector<std::uint16_t> sums) const;
    // Passes consume the sums of consecutive unit ranges ending at ends, each from its start.
    using SumsStream = std::function<void(const std::vector<std::size_t>& ends, const std::function<void(const std::uint16_t*)>& consume)>;
    // The same from streamed sums, symmetrized already, in ranges of whole decisions of at most
    // chunkUnits units unless a single decision exceeds it, plus any decisions after them that have
    // no board-compatible hands.
    StrategySnapshot ExportStrategy(std::size_t chunkUnits, const SumsStream& stream) const;
    // Compare states by these values: re-encoding can store equal values differently.
    TrainingState Decode(const QuantizedState& state) const;

private:
    void PrepareHandImages();
    // SymmetrizeSums of one decision's units over target, which holds them already.
    void SymmetrizeDecision(const Node& node, const std::uint16_t* units, std::uint16_t* target) const;
    void PrepareChanceTasks();
    void PrepareRunoutOutcomes();
    void PrepareLocks(const std::vector<NodeLock>& locks);
};

// One player update's DCFR weights (see DcfrSession::Run). Positive regrets are stored divided
// by positiveScale, the product of the positive discounts of the player's earlier updates, so
// pruned updates need no discount pass; negative regrets decay lazily over skipped updates
// (gpu::NegativeDecay).
struct UpdateWeights
{
    std::uint32_t update;   // 1-based index of this update for the updating player
    float positiveScale;    // product of earlier positive discounts
    float positiveInverse;  // 1 / positiveScale
    float averageWeight;    // t^3, the weight of this update's reach * policy in the sums
    float negativeDiscount; // factor of a negative regret after this update
};
} // namespace solver::engine
