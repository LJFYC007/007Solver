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
    WorkBuffer,
    OutcomesBuffer,
    RegretsBuffer,
    SumsBuffer,
    ScratchBuffer,
    ValuesBuffer,
    StateBuffer,
    PassBuffer,
    kDataBufferCount = StateBuffer,
    kBufferCount = PassBuffer,
};

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
    U32 edge;
    U32 count;
    U32 actor;
    NodeKind kind;
    U32 rankRow;
    U32 rankCounts[2];
    U32 outcomeRow; // zero for flop, card index + 1 for turn
    U32 dealtCard;
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
    U32 hands[2];
    U32 stride;
    U32 player;
    U32 evaluation;
    U32 outcomeRows;
    float positiveDiscount;
    float averageDiscount;
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
};
} // namespace gpu
} // namespace engine
} // namespace solver
