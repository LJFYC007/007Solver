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
    OutcomesBuffer,
    RegretsBuffer,
    SumsBuffer,
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
    kGroupFlags = 8, // leading Terminal group memory floats for the group-wide reach test: one per SIMD group
    kNoIndex = 0xffffffffu,
};
// Pass::sync bits ordering the two lanes; a sequential executor may ignore them.
enum : U32
{
    kForkAfter = 1,  // lane 1 work emitted later depends on this lane-0 pass
    kWaitFork = 2,   // this lane-1 pass waits for the latest fork
    kJoinBefore = 4, // this lane-0 pass waits for all lane-1 work emitted so far
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
    U32 rankRow;
    U32 rankCounts[2];
    U32 outcomeRow;   // zero for flop, card index + 1 for turn
    U32 stamp;        // index of this node's update stamp
    U32 childSlot[3]; // leading entries of childSlots, so small decisions skip that lookup
    float utility[3];
};
struct Hand
{
    U64 mask;
    float weight;
    float divisor;
    U32 card0;
    U32 card1;
    int matching;
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
    U32 padding;
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
    U32 lanes;    // threads per work item for one-dimensional launches; LaunchPass sets Reach's
    U32 boundary; // Reach items derive the opponent's reach from the game root or a dealt card
    U32 split;    // deeper Reach items before this index belong to actor 0, the rest to actor 1
    U32 lane;     // river batches alternate lanes with disjoint scratch so their passes can overlap
    U32 sync;     // kForkAfter, kWaitFork and kJoinBefore bits
};
} // namespace gpu
} // namespace engine
} // namespace solver
