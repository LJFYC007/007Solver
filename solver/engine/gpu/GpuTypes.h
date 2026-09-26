#pragma once

// This POD layout is also compiled for the device by CUDA, so it holds no host pointers.
#if defined(__CUDACC__)
#define GPU_TYPES_INLINE __host__ __device__ __forceinline__
#else
#define GPU_TYPES_INLINE inline
#endif
namespace solver::engine::gpu
{
using U32 = unsigned int;
using U64 = unsigned long long;
// Order of the kernels' buffer arguments, shared by host allocation.
enum BufferIndex : U32
{
    NodesBuffer,
    HandsBuffer,
    RanksBuffer,
    RunoutsBuffer,
    OrderBuffer,
    CardsBuffer,
    OutcomesBuffer, // runout win/loss counts per hand pair: player-0-major rows, then player-1-major copies
    RegretsBuffer,  // int16 units in the HandTraversalData::StateUnits layout (see GpuQuantize.h)
    SumsBuffer,     // uint16 units in the same layout
    ScratchBuffer, // per slot one row of stride floats (a multiple of four): the opponent's reach entering its node, then the slot's values
    FlagsBuffer,   // per slot: whether the subtree carries opponent reach
    StampsBuffer,  // two halves of stampCount node stamps, alternating by update parity
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
    kTileGroup = 256,     // most threads per Reach or Backup hand tile, which their launch bounds and Reach's compaction assume
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

// One work item's 64-byte record, 16-byte aligned so a thread loads it as four 16-byte vectors.
struct alignas(16) Node
{
    U64 strategy;
    // The node's board mask, which Terminal tests against each hand. A decision's Backup record
    // instead holds in its low and high halves, and in fold, per action, the child there if the
    // decision's Backup inlines it (see Plan; only decisions with two or three children do): its
    // link | its count << 30, else kNoIndex.
    U64 board;
    U32 slot;
    // Slot holding each player's reach entering this node when that player is the
    // opponent of the update. It was written by the nearest ancestor that changed it
    // (that player's decision or a chance node), so siblings share it with their parent.
    U32 reachSlot[2];
    // The first child's slot; a node's children occupy consecutive slots, which every kernel
    // relies on. A showdown with a fold sibling instead holds its parent's slot (see fold).
    U32 link;
    // The node's kind, actor, whether it is the game root, its child count and its row, packed
    // by PackInfo: a showdown's row is the rank row of its board, a forced runout's zero for
    // flop or card index + 1 for turn.
    U32 info;
    U32 rankCounts; // Showdown: legal hands of player 0 | player 1 << 16 in its rank row
    U32 stamp;      // index of this node's update stamp
    // Showdown with a Fold sibling: that fold's slot, whose values this showdown's Terminal
    // block writes, and its player-0 utility; kNoIndex without one. Such a showdown's strategy,
    // count, actor and stamp are its parent's, fields a leaf does not use otherwise, its link
    // is the parent's slot, and the fold is action 0 when its slot precedes the showdown's;
    // when the two are the parent's only children, the block also backs the parent up. A region
    // root's boundary Reach record, and the Reach record of a child of a root whose boundary
    // pass a player skips (see Plan), instead holds the root's chance parent's reachSlot, the
    // dealt card as board and the outcome probability 1 / (parent count - 4) as foldUtility;
    // other decision records hold a zero foldUtility.
    U32 fold;
    float foldUtility;
    // Player 0's win, tie and loss payoffs. A forced runout's win and loss instead hold the
    // payoffs' differences from the tie per runout, which scale its win and loss counts; as
    // negation is exact, player 1's scales are the negated loss and win entries.
    float utility[3];
};
// Node::info: kind in 3 bits, actor in 1, the root flag in 1, the child count in 6 (chance
// nodes deal at most 49 cards) and the row above.
enum : U32
{
    kInfoCountBits = 6,
    kInfoRowShift = 5 + kInfoCountBits,
};
GPU_TYPES_INLINE U32 PackInfo(NodeKind kind, U32 actor, bool root, U32 count, U32 row)
{
    return U32(kind) | actor << 3 | (root ? 1u : 0u) << 4 | count << 5 | row << kInfoRowShift;
}
GPU_TYPES_INLINE NodeKind Kind(U32 info)
{
    return NodeKind(info & 7u);
}
GPU_TYPES_INLINE U32 Actor(U32 info)
{
    return (info >> 3) & 1u;
}
GPU_TYPES_INLINE bool IsRoot(U32 info)
{
    return ((info >> 4) & 1u) != 0;
}
GPU_TYPES_INLINE U32 Count(U32 info)
{
    return (info >> 5) & ((1u << kInfoCountBits) - 1);
}
GPU_TYPES_INLINE U32 Row(U32 info)
{
    return info >> kInfoRowShift;
}
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
// 16-byte aligned and padded so a block loads it as four 16-byte vectors.
struct alignas(16) State
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
    U32 orderSection;    // offset of player 1's section in a rank row of the order buffer (see Plan::order)
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
