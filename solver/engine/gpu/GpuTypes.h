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
struct Node
{
    U64 strategy;
    U64 board;
    U32 parent;
    U32 action;
    U32 slot;
    U32 edge;
    U32 count;
    U32 actor;
    U32 kind; // decision, chance, fold, showdown, forced runout
    U32 rankRow;
    U32 rankCounts[2];
    U32 outcomeRow; // zero for flop, card index + 1 for turn
    U32 dealtCard;
    float utility[3];
    U32 padding;
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
    Prefix,
    Terminal,
    Backup,
    Outcomes,
};
struct Pass
{
    Kernel kernel;
    U32 offset;
    U32 count;
};
} // namespace gpu
} // namespace engine
} // namespace solver
