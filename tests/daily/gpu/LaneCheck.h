#pragma once
// Static check of the two-lane CUDA schedule. Rebuilds the happens-before graph that
// CudaExecutor::Launch creates from Pass::sync during stream capture, derives each pass's
// read/write footprint from the kernels in GpuKernels.inc, and reports
//   - pairs of passes that the graph leaves unordered but that touch the same data, and
//   - capture errors: lane-1 work before lane_ joins the capture, a wait on a fork not
//     recorded in this capture, or lane-1 work not joined back before capture ends.
// Emulated schedules only replay legal orders, so they cannot see overlapping lane
// scratch; this check can.

#include "engine/gpu/GpuPlan.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace emu
{
using namespace solver::engine::gpu;

struct LaneCheckResult
{
    std::size_t unorderedPairs = 0;
    std::vector<std::string> errors;
};

namespace lane_check_detail
{
enum Resource : std::uint64_t
{
    Scratch = 1,
    Values,
    Flags,
    Regrets,
    Sums,
    Stamps,
};
inline std::uint64_t Key(Resource resource, std::uint64_t index)
{
    return (std::uint64_t(resource) << 56) | index;
}
struct Footprint
{
    std::vector<std::uint64_t> reads, writes;
    void Finish()
    {
        for (auto* list : {&reads, &writes})
        {
            std::sort(list->begin(), list->end());
            list->erase(std::unique(list->begin(), list->end()), list->end());
        }
    }
};
inline bool Intersect(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b)
{
    for (auto i = a.begin(), j = b.begin(); i != a.end() && j != b.end();)
        if (*i < *j)
            ++i;
        else if (*j < *i)
            ++j;
        else
            return true;
    return false;
}

// Mirrors the Reach, Terminal and Backup kernels' buffer indexing for updating player `player`.
inline Footprint PassFootprint(const Plan& plan, const Pass& launched, U32 player)
{
    Footprint f;
    const U32 opponent = 1 - player;
    const auto children = [&](const Node& n, auto&& visit)
    {
        for (U32 a = 0; a < n.count; ++a)
            visit(plan.childSlots[n.edge + a]);
    };
    for (U32 i = 0; i < launched.count; ++i)
    {
        const Node& n = plan.nodes[launched.offset + i];
        switch (launched.operation)
        {
        case Kernel::Reach:
        {
            bool propagates = true;
            if (launched.boundary)
            {
                if (n.parent != kNoIndex)
                    f.reads.push_back(Key(Scratch, plan.nodes[n.parent].reachSlot[opponent]));
                f.writes.push_back(Key(Scratch, n.slot));
                propagates = n.kind == NodeKind::Decision && n.actor == opponent;
            }
            else
                f.reads.push_back(Key(Scratch, n.reachSlot[opponent]));
            if (propagates)
            {
                children(n, [&](U32 slot) { f.writes.push_back(Key(Scratch, slot)); });
                f.reads.push_back(Key(Regrets, n.strategy)); // entry policy (sums while evaluating)
                f.writes.push_back(Key(Sums, n.strategy));
            }
            break;
        }
        case Kernel::Terminal:
            f.reads.push_back(Key(Scratch, n.reachSlot[opponent]));
            f.writes.push_back(Key(Values, n.slot));
            f.writes.push_back(Key(Flags, n.slot));
            break;
        case Kernel::Backup:
            children(n,
                     [&](U32 slot)
                     {
                         f.reads.push_back(Key(Flags, slot));
                         f.reads.push_back(Key(Values, slot));
                     });
            f.writes.push_back(Key(Flags, n.slot));
            f.writes.push_back(Key(Values, n.slot));
            if (n.kind == NodeKind::Decision && n.actor == player)
            {
                f.reads.push_back(Key(Regrets, n.strategy));
                f.writes.push_back(Key(Regrets, n.strategy));
                f.reads.push_back(Key(Stamps, n.stamp));
                f.writes.push_back(Key(Stamps, n.stamp));
            }
            break;
        case Kernel::Outcomes:
            break;
        }
    }
    f.Finish();
    return f;
}
} // namespace lane_check_detail

// Checks one player's captured graph.
inline LaneCheckResult CheckLanes(const Plan& plan, U32 player)
{
    using namespace lane_check_detail;
    LaneCheckResult result;
    const auto count = plan.passes.size();
    // Stream frontiers: the ops the next launch or event record on that stream depends on.
    std::vector<std::size_t> mainFrontier, laneFrontier, fork;
    bool laneCaptured = false, forkRecorded = false;
    std::vector<std::vector<std::size_t>> predecessors(count);
    std::vector<std::size_t> laneWork; // lane-1 passes not yet joined into the main stream
    for (std::size_t i = 0; i < count; ++i)
    {
        const Pass& pass = plan.passes[i];
        if (pass.sync & kWaitFork)
        {
            if (!forkRecorded)
                result.errors.push_back("pass " + std::to_string(i) + " waits on a fork not recorded in this capture");
            laneFrontier.insert(laneFrontier.end(), fork.begin(), fork.end());
            laneCaptured = true;
        }
        if (pass.sync & kJoinBefore)
        {
            mainFrontier.insert(mainFrontier.end(), laneFrontier.begin(), laneFrontier.end());
            laneWork.clear();
        }
        if (pass.count)
        {
            if (pass.lane && !laneCaptured)
                result.errors.push_back("lane-1 pass " + std::to_string(i) + " launches before lane_ joins the capture");
            auto& frontier = pass.lane ? laneFrontier : mainFrontier;
            predecessors[i] = frontier;
            frontier.assign(1, i);
            if (pass.lane)
                laneWork.push_back(i);
        }
        if (pass.sync & kForkAfter)
        {
            fork = mainFrontier;
            forkRecorded = true;
        }
    }
    if (!laneWork.empty())
        result.errors.push_back(std::to_string(laneWork.size()) + " lane-1 passes are never joined before capture ends");

    // Reachability in emission order, which is topological.
    const std::size_t words = (count + 63) / 64;
    std::vector<std::uint64_t> reach(count * words, 0);
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t p : predecessors[i])
        {
            reach[i * words + p / 64] |= std::uint64_t(1) << (p % 64);
            for (std::size_t w = 0; w < words; ++w)
                reach[i * words + w] |= reach[p * words + w];
        }
    std::vector<Footprint> footprints(count);
    for (std::size_t i = 0; i < count; ++i)
        if (plan.passes[i].count)
            footprints[i] = PassFootprint(plan, LaunchPass(plan.passes[i], plan.state, player), player);
    for (std::size_t j = 0; j < count; ++j)
        for (std::size_t i = 0; i < j; ++i)
        {
            if (!plan.passes[i].count || !plan.passes[j].count || (reach[j * words + i / 64] >> (i % 64) & 1))
                continue;
            ++result.unorderedPairs;
            const auto& a = footprints[i];
            const auto& b = footprints[j];
            if (Intersect(a.writes, b.writes) || Intersect(a.writes, b.reads) || Intersect(a.reads, b.writes))
                result.errors.push_back("unordered passes " + std::to_string(i) + " (lane " + std::to_string(plan.passes[i].lane) + ") and " +
                                        std::to_string(j) + " (lane " + std::to_string(plan.passes[j].lane) + ") touch the same data");
        }
    return result;
}
} // namespace emu
