#pragma once

// This POD layout is also compiled by CUDA and Metal. No host pointers or size_t.
namespace solver
{
namespace engine
{
namespace gpu
{
using U32 = unsigned int;
#ifdef __METAL_VERSION__
using U64 = ulong;
#else
using U64 = unsigned long long;
#endif
// Binding order shared by host allocation and both kernel compilers.
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
    ScratchBuffer,
    ValuesBuffer,
    FlagsBuffer,  // per slot: whether the subtree carries opponent reach
    StampsBuffer, // two halves of stampCount node stamps, alternating by update parity
    StateBuffer,
    PassBuffer,
    kDataBufferCount = StateBuffer,
    kBufferCount = PassBuffer,
};
// Each player's card list starts with 52 card offsets and an end offset into the
// cards buffer; hand indices follow both players' offsets.
enum : U32
{
    kCardListStride = 53,
    kCardListHeader = 2 * kCardListStride,
    kGroupFlags = 8,  // leading Terminal group memory floats for the group-wide reach test: one per SIMD group
    kLaneCount = 3,   // Pass::lane values: two alternating leaf-batch lanes, then the spine
    kMaxActions = 16, // actions per decision the kernels' per-hand entry arrays hold
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
    U32 parent;
    U32 slot;
    // Slot holding each player's reach entering this node when that player is the
    // opponent of the update. It was written by the nearest ancestor that changed it
    // (that player's decision or a chance node), so siblings share it with their parent.
    U32 reachSlot[2];
    U32 edge;
    U32 count;
    U32 actor;
    NodeKind kind;
    U32 row;        // Showdown: rank row of its board; ForcedRunout: zero for flop, card index + 1 for turn
    U32 rankCounts; // Showdown: legal hands of player 0 | player 1 << 16 in its rank row
    U32 stamp;      // index of this node's update stamp
    // Showdown with a Fold sibling: that fold's slot, whose values this showdown's Terminal
    // block writes, and its player-0 utility; kNoIndex without one. When the two are the
    // parent's only children, the block also backs the parent up and parent names the
    // parent's record after the work items (see Plan::nodes).
    U32 fold;
    float foldUtility;
    // Leading entries of childSlots, so small decisions skip that lookup; a chance node's
    // children occupy consecutive slots from childSlot[0].
    U32 childSlot[3];
    float utility[3];
};
// Two 16-byte halves: Terminal loads the first for every hand and the second only at
// showdowns; Reach reads the weight alone.
struct Hand
{
    U64 mask;
    float divisor;
    U32 cards; // card0 | card1 << 8 | (identical opponent hand + 1) << 16, zero without one
    float weight;
    U32 runs[2]; // per held card, the opponent's holder run: Terminal run-table offset | length << 16
    U32 padding;
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
    U32 orderPitch;      // entries per rank row of the order buffer, even so hand pairs align
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
    U32 lanes;    // threads per work item for one-dimensional launches; LaunchPass sets Reach's and Backup's
    U32 boundary; // Reach items derive the opponent's reach from the game root or a dealt card
    U32 split;    // deeper Reach items before this index belong to actor 0, the rest to actor 1
    U32 lane;     // the stream running this pass; a sequential executor runs the pass list in order
};
} // namespace gpu
} // namespace engine
} // namespace solver
