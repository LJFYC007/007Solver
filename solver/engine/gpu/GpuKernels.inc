// Shared FP32 arithmetic, compiled as CUDA C++ or Metal Shading Language.
using namespace solver::engine::gpu;
#ifdef __METAL_VERSION__
#define GPU_INLINE inline
#define GPU_DEVICE device
#define GPU_BUFFER(type, name, index) device type* name [[buffer(index)]]
#define GPU_ENTRY(name) kernel void name
#define GPU_TAIL constant State &state [[buffer(13)]], constant Pass &pass [[buffer(14)]], uint gid [[thread_position_in_grid]]
#define GPU_INDEX gid
#else
#define GPU_INLINE __device__ __forceinline__
#define GPU_DEVICE
#define GPU_BUFFER(type, name, index) type* name
#define GPU_ENTRY(name) __global__ void name
#define GPU_TAIL const State *statePointer, Pass pass
#define GPU_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#endif
#define GPU_ARGUMENTS                                                                                                                 \
    GPU_BUFFER(const Node, nodes, 0), GPU_BUFFER(const U32, edges, 1), GPU_BUFFER(const Hand, hands, 2),                              \
        GPU_BUFFER(const unsigned short, ranks, 3), GPU_BUFFER(const int, runouts, 4), GPU_BUFFER(const U32, order, 5),               \
        GPU_BUFFER(const U32, cards, 6), GPU_BUFFER(const U32, work, 7), GPU_BUFFER(U32, outcomes, 8), GPU_BUFFER(float, regrets, 9), \
        GPU_BUFFER(float, sums, 10), GPU_BUFFER(float, scratch, 11), GPU_BUFFER(float, values, 12), GPU_TAIL
#ifdef __METAL_VERSION__
#define GPU_STATE
#else
#define GPU_STATE const State state = *statePointer;
#endif

GPU_INLINE float Policy(
    Node n,
    U32 hand,
    U32 action,
    U32 count,
    bool average,
    GPU_DEVICE const float* regrets,
    GPU_DEVICE const float* sums
)
{
    float total = 0.0f;
    for (U32 a = 0; a < n.count; ++a)
    {
        const float x = average ? sums[n.strategy + U64(a) * count + hand] : regrets[n.strategy + U64(a) * count + hand];
        total += x > 0.0f ? x : 0.0f;
    }
    const float x = average ? sums[n.strategy + U64(action) * count + hand] : regrets[n.strategy + U64(action) * count + hand];
    return total > 0.0f ? (x > 0.0f ? x : 0.0f) / total : 1.0f / float(n.count);
}

GPU_ENTRY(Reach)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 index = GPU_INDEX;
    const U32 item = index / state.stride, h = index % state.stride;
    if (item >= pass.count)
        return;
    const Node n = nodes[work[pass.offset + item]];
    const U32 total = state.hands[0] + state.hands[1];
    for (U32 p = 0; p < 2; ++p)
    {
        if (h >= state.hands[p])
            continue;
        // Own reach is consumed only by training updates at nonterminal nodes.
        if (p == state.player && (state.evaluation || n.kind >= 2))
            continue;
        const U32 base = p ? state.hands[0] : 0;
        const Hand hand = hands[base + h];
        float value;
        if (n.parent == 0xffffffffu)
            value = p == state.player ? 1.0f : hand.weight;
        else
        {
            const Node parent = nodes[n.parent];
            value = scratch[U64(parent.slot) * total + base + h];
            if (parent.kind == 1)
                value = hand.mask & (U64(1) << n.dealtCard) ? 0.0f : value * (p == state.player ? 1.0f : 1.0f / float(parent.count - 4));
            else if (parent.actor == p)
                value *= Policy(parent, h, n.action, state.hands[p], state.evaluation != 0, regrets, sums);
        }
        scratch[U64(n.slot) * total + base + h] = value;
    }
}

// One independent ordered sweep per terminal. Prefix and suffix are accumulated
// separately; loss mass never subtracts two nearly equal cumulative totals.
GPU_ENTRY(Prefix)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 item = GPU_INDEX;
    if (item >= pass.count)
        return;
    const Node n = nodes[work[pass.offset + item]];
    if (n.kind == 4)
        return;
    const U32 p = 1 - state.player, count = state.hands[p];
    const U32 total = state.hands[0] + state.hands[1], base = p ? state.hands[0] : 0;
    GPU_DEVICE const float* reach = scratch + U64(n.slot) * total + base;
    GPU_DEVICE float* prefix = values + U64(n.slot) * 3 * state.stride + state.stride;
    float sum = 0.0f;
    if (n.kind == 2)
    {
        for (U32 h = 0; h < count; ++h)
            sum += reach[h];
        prefix[0] = sum;
        return;
    }
    GPU_DEVICE const U32* sorted = order + U64(n.rankRow) * 2 * total + base;
    for (U32 i = 0; i < n.rankCounts[p]; ++i)
    {
        sum += reach[sorted[i]];
        prefix[i] = sum;
    }
    sum = 0.0f;
    for (U32 i = n.rankCounts[p]; i-- > 0;)
    {
        sum += reach[sorted[i]];
        prefix[state.stride + i] = sum;
    }
}

GPU_ENTRY(Terminal)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 index = GPU_INDEX, item = index / state.stride, h = index % state.stride;
    const U32 p = state.player, opponent = 1 - p;
    if (item >= pass.count || h >= state.hands[p])
        return;
    const Node n = nodes[work[pass.offset + item]];
    const U32 total = state.hands[0] + state.hands[1], mineBase = p ? state.hands[0] : 0, oppBase = p ? 0 : state.hands[0];
    const Hand mine = hands[mineBase + h];
    GPU_DEVICE const float* reach = scratch + U64(n.slot) * total + oppBase;
    GPU_DEVICE float* output = values + U64(n.slot) * 3 * state.stride;
    GPU_DEVICE const float* prefix = output + state.stride;
    if ((mine.mask & n.board) || mine.divisor <= 0.0f)
    {
        output[h] = 0.0f;
        return;
    }
    const float win = p == 0 ? n.utility[0] : -n.utility[2];
    const float tie = p == 0 ? n.utility[1] : -n.utility[1];
    const float loss = p == 0 ? n.utility[2] : -n.utility[0];
    float mass = 0.0f, wins = 0.0f, losses = 0.0f;
    if (n.kind == 4)
    {
        for (U32 other = 0; other < state.hands[opponent]; ++other)
        {
            if (mine.mask & hands[oppBase + other].mask)
                continue;
            const U64 pair = p == 0 ? U64(h) * state.hands[1] + other : U64(other) * state.hands[1] + h;
            const U32 counts = outcomes[U64(n.outcomeRow) * state.hands[0] * state.hands[1] + pair];
            mass += reach[other];
            wins += reach[other] * float(p == 0 ? counts & 0xffffu : counts >> 16);
            losses += reach[other] * float(p == 0 ? counts >> 16 : counts & 0xffffu);
        }
        const float divisor = n.outcomeRow == 0 ? 990.0f : 44.0f;
        output[h] = (tie * mass + (win - tie) * (wins / divisor) + (loss - tie) * (losses / divisor)) / mine.divisor;
        return;
    }
    const U32 rank = n.kind == 3 ? ranks[U64(n.rankRow) * total + mineBase + h] : 0;
    const U32 legal = n.kind == 3 ? n.rankCounts[opponent] : state.hands[opponent];
    if (n.kind == 3)
    {
        // Rank boundaries depend only on the board and ranges, not terminal reach.
        const U32 bounds = order[U64(n.rankRow) * 2 * total + total + mineBase + h];
        const U32 lower = bounds & 0xffffu, upper = bounds >> 16;
        wins = lower ? prefix[lower - 1] : 0.0f;
        losses = upper < legal ? prefix[state.stride + upper] : 0.0f;
    }
    const float totalMass = n.kind == 2 ? prefix[0] : legal ? prefix[legal - 1] : 0.0f;
    mass = totalMass;
    const float winTotal = wins, lossTotal = losses;
    float blocked = 0.0f, blockedWins = 0.0f, blockedLosses = 0.0f;
    for (U32 c = 0; c < 2; ++c)
    {
        const U32 card = c == 0 ? mine.card0 : mine.card1;
        for (U32 entry = cards[opponent * 53 + card]; entry < cards[opponent * 53 + card + 1]; ++entry)
        {
            const U32 other = cards[entry];
            const float r = reach[other];
            blocked += r;
            if (n.kind == 3)
            {
                const U32 otherRank = ranks[U64(n.rankRow) * total + oppBase + other];
                if (otherRank < rank)
                    blockedWins += r;
                if (otherRank > rank)
                    blockedLosses += r;
            }
        }
    }
    mass -= blocked;
    if (mine.matching >= 0)
        mass += reach[mine.matching];
    wins -= blockedWins;
    losses -= blockedLosses;
    // Recompute tiny compatible masses directly, including on FP32 hardware.
    const bool repairMass = mass < 0.0001f * totalMass;
    const bool repairWins = wins < 0.0001f * winTotal;
    const bool repairLosses = losses < 0.0001f * lossTotal;
    if (repairMass || repairWins || repairLosses)
    {
        float m = 0.0f, w = 0.0f, l = 0.0f;
        for (U32 other = 0; other < state.hands[opponent]; ++other)
        {
            if (mine.mask & hands[oppBase + other].mask)
                continue;
            const float r = reach[other];
            m += r;
            if (n.kind == 3)
            {
                const U32 otherRank = ranks[U64(n.rankRow) * total + oppBase + other];
                if (otherRank < rank)
                    w += r;
                if (otherRank > rank)
                    l += r;
            }
        }
        if (repairMass)
            mass = m;
        if (repairWins)
            wins = w;
        if (repairLosses)
            losses = l;
    }
    output[h] = n.kind == 2 ? (mass / mine.divisor) * (p == 0 ? n.utility[0] : -n.utility[0])
                            : (tie * mass + (win - tie) * wins + (loss - tie) * losses) / mine.divisor;
}

GPU_ENTRY(Backup)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 index = GPU_INDEX, item = index / state.stride, h = index % state.stride;
    if (item >= pass.count || h >= state.hands[state.player])
        return;
    const Node n = nodes[work[pass.offset + item]];
    const bool acting = n.kind == 0 && n.actor == state.player;
    // Reuse the entry-policy denominator for value backup and strategy sums.
    float positive = 0.0f;
    if (acting && !state.evaluation)
        for (U32 a = 0; a < n.count; ++a)
        {
            const float r = regrets[n.strategy + U64(a) * state.hands[n.actor] + h];
            positive += r > 0.0f ? r : 0.0f;
        }
    float value = 0.0f;
    for (U32 a = 0; a < n.count; ++a)
    {
        const float child = values[U64(nodes[edges[n.edge + a]].slot) * 3 * state.stride + h];
        if (acting && state.evaluation)
            value = a == 0 || child > value ? child : value;
        else if (acting)
        {
            const float r = regrets[n.strategy + U64(a) * state.hands[n.actor] + h];
            const float probability = positive > 0.0f ? (r > 0.0f ? r : 0.0f) / positive : 1.0f / float(n.count);
            value += child * probability;
        }
        else
            value += child;
    }
    values[U64(n.slot) * 3 * state.stride + h] = value;
    if (acting && !state.evaluation)
    {
        const U32 total = state.hands[0] + state.hands[1], base = state.player ? state.hands[0] : 0;
        const float own = scratch[U64(n.slot) * total + base + h];
        for (U32 a = 0; a < n.count; ++a)
        {
            const U64 entry = n.strategy + U64(a) * state.hands[n.actor] + h;
            const float old = regrets[entry];
            const float probability = positive > 0.0f ? (old > 0.0f ? old : 0.0f) / positive : 1.0f / float(n.count);
            const float child = values[U64(nodes[edges[n.edge + a]].slot) * 3 * state.stride + h];
            const float r = old + (child - value);
            regrets[entry] = r * (r > 0.0f ? state.positiveDiscount : 0.5f);
            sums[entry] = state.averageDiscount * (sums[entry] + own * probability);
        }
    }
}

GPU_ENTRY(Outcomes)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U64 index = GPU_INDEX, pairs = U64(state.hands[0]) * state.hands[1];
    if (index >= pairs * (pass.offset ? 1 : state.outcomeRows))
        return;
    const U32 row = U32(index / pairs), h0 = U32(index % pairs / state.hands[1]), h1 = U32(index % state.hands[1]);
    const U64 mask0 = hands[h0].mask, mask1 = hands[state.hands[0] + h1].mask;
    const U64 board = nodes[0].board;
    U32 wins = 0, losses = 0;
    if (!(mask0 & mask1))
    {
        if (pass.offset)
        {
            for (U32 r = 1; r < state.outcomeRows; ++r)
            {
                const U32 result = outcomes[U64(r) * pairs + index];
                wins += result & 0xffffu;
                losses += result >> 16;
            }
            wins /= 2;
            losses /= 2;
        }
        else if (row || state.outcomeRows == 1)
        {
            const U64 excluded = board | mask0 | mask1;
            const U32 firstBegin = row ? row - 1 : 0, firstEnd = row ? row : 52;
            for (U32 first = firstBegin; first < firstEnd; ++first)
            {
                if (excluded & (U64(1) << first))
                    continue;
                for (U32 second = 0; second < (row ? 52 : first); ++second)
                {
                    if (first == second || (excluded & (U64(1) << second)))
                        continue;
                    const U32 high = first > second ? first : second, low = first < second ? first : second;
                    const int rankRow = runouts[high * (high - 1) / 2 + low];
                    const U64 offset = U64(rankRow) * (state.hands[0] + state.hands[1]);
                    const U32 a = ranks[offset + h0], b = ranks[offset + state.hands[0] + h1];
                    wins += a > b;
                    losses += a < b;
                }
            }
        }
    }
    outcomes[index] = wins | (losses << 16);
}
