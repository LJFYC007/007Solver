#pragma once

// This POD layout is also compiled for the device by CUDA, so it holds no host pointers.
namespace solver::engine::gpu
{
using U32 = unsigned int;
using U64 = unsigned long long;
// Order of the kernels' buffer arguments, shared by host allocation.
enum BufferIndex : U32
{
    NodesBuffer,
    ChildSlotsBuffer,
    HandsBuffer,
    RanksBuffer,
    RunoutsBuffer,
    OrderBuffer,
    CardsBuffer,
    OutcomesBuffer, // runout win/loss counts per hand pair: player-0-major rows, then player-1-major copies
    RegretsBuffer,  // int16 units in the HandTraversalData::StateUnits layout (see GpuQuantize.h)
    SumsBuffer,     // uint16 units in the same layout
    ScratchBuffer,  // per slot one row of stride floats: the opponent's reach entering its node, then the slot's values
    FlagsBuffer,    // per slot: whether the subtree carries opponent reach
    StampsBuffer,   // two halves of stampCount node stamps, alternating by update parity
    StateBuffer,
    kBufferCount,
};
// Each player's card list starts with 52 card offsets and an end offset into the
// cards buffer; hand indices follow both players' offsets.
enum : U32
{
    kCardListStride = 53,
    kCardListHeader = 2 * kCardListStride,
    kLaneCount = 4,       // Pass::lane values: three leaf-batch lanes taken in turn, then the spine
    kMaxActions = 16,     // actions per decision the kernels' per-hand entry arrays hold
    kWarpSize = 32,       // threads per warp
    kTerminalGroup = 128, // threads per Terminal group, which the kernel's launch bounds and warp roles assume
    kNoIndex = 0xffffffffu,
};
// Also the CPU traversal node kind. Fold, Showdown and ForcedRunout are leaves.
enum class NodeKind : U32
{
    Decision,
    Chance,
    Fold,
    Showdown,
    ForcedRunout,
};

struct Node
{
    U64 strategy;
    U64 board;
    U32 parent; // traversal index; kernels only test for kNoIndex, the game root
    U32 slot;
    // Slot holding each player's reach entering this node when that player is the
    // opponent of the update. It was written by the nearest ancestor that changed it
    // (that player's decision or a chance node), so siblings share it with their parent.
    U32 reachSlot[2];
    U32 edge;
    U32 count;
    U32 actor;
    NodeKind kind;
    // A decision's Backup record instead holds in row, rankCounts and fold, per action, the child
    // there if the decision's Backup inlines it (see Plan; only decisions with two or three
    // children do): its childSlot[0] | its count << 30, else kNoIndex.
    U32 row;        // Showdown: rank row of its board; ForcedRunout: zero for flop, card index + 1 for turn
    U32 rankCounts; // Showdown: legal hands of player 0 | player 1 << 16 in its rank row
    U32 stamp;      // index of this node's update stamp
    // Showdown with a Fold sibling: that fold's slot, whose values this showdown's Terminal
    // block writes, and its player-0 utility; kNoIndex without one. Such a showdown's strategy,
    // count, actor, stamp and childSlot[0] are its parent's, fields a leaf does not use
    // otherwise, and its edge is the parent's slot; when the two are the parent's only
    // children, the block also backs the parent up. A region root's boundary Reach record,
    // and the Reach record of a child of a root whose boundary pass a player skips (see
    // Plan), instead holds the root's chance parent's reachSlot, the dealt card as
    // board and the outcome probability 1 / (parent count - 4) as foldUtility; other decision
    // records hold a zero foldUtility.
    U32 fold;
    float foldUtility;
    // Leading entries of childSlots, so small decisions skip that lookup; a node's children
    // occupy consecutive slots from childSlot[0].
    U32 childSlot[3];
    // Player 0's win, tie and loss payoffs. A forced runout's win and loss instead hold the
    // payoffs' differences from the tie per runout, which scale its win and loss counts; as
    // negation is exact, player 1's scales are the negated loss and win entries.
    float utility[3];
};
// Two 16-byte halves, so Terminal loads the first for every hand in one access; Reach reads
// single fields.
struct Hand
{
    U64 mask;
    float scale; // ValueScale of the hand's compatible opponent mass (see HandEvaluation.h)
    U32 cards;   // card0 | card1 << 8 | (identical opponent hand + 1) << 16, zero without one
    float weight;
    U32 padding[3];
};
struct State
{
    U64 board; // initial public board mask
    U32 hands[2];
    U32 stride;
    U32 player;
    U32 evaluation;
    U32 outcomeRows;
    U32 update;     // 1-based index of this update for the updating player
    U32 stampCount; // node stamps per half of the stamps buffer
    // UpdateWeights: positive regrets are stored divided by positiveScale.
    float positiveScale;
    float positiveInverse;
    float averageWeight; // t^2 weight of this update's reach * policy in the strategy sums
    U32 orderPitch;      // entries per rank row of the order buffer, a multiple of four so hand entries align
};
enum class Kernel : U32
{
    Reach,
    Terminal,
    Backup,
    Outcomes,
};
enum class OutcomeStage : U32
{
    None,
    CountRunouts,
    SumTurns,
};
struct Pass
{
    Kernel operation;
    U32 offset;
    U32 count;
    OutcomeStage outcomeStage;
    // Threads per work item; LaunchPass sets those of Plan::passes, Terminal's being the opponent
    // hands its shared memory holds.
    U32 lanes;
    U32 lane; // the stream running this pass
    // In Plan::passes, player p's updates launch items begin[p] to end[p] (see Plan and
    // LaunchPass); initialization passes launch all count items.
    U32 begin[2];
    U32 end[2];
};
} // namespace solver::engine::gpu
