// Prints a scenario's GPU plan shape and checks its two-lane CUDA schedule (LaneCheck.h) for
// both updating players. When the plan uses lane 1, it also mutates copies of the plan (no
// joins, no fork waits, overlapping lane scratch) and requires the check to catch each one.
// Usage: daily_planstats <scenario.json>; exits 1 on any lane error or undetected mutation.
#include "LaneCheck.h"
#include "engine/HandTraversalData.h"
#include "engine/gpu/GpuPlan.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include <cstdio>
#include <functional>

using namespace solver;
using namespace solver::engine::gpu;

namespace
{
std::size_t Errors(const Plan& plan)
{
    return emu::CheckLanes(plan, 0).errors.size() + emu::CheckLanes(plan, 1).errors.size();
}

// Returns true when the mutation applied and the check reported it.
bool Detected(const Plan& plan, const char* name, const std::function<bool(Plan&)>& mutate)
{
    Plan mutated = plan;
    if (!mutate(mutated))
    {
        std::printf("mutation %s: not applicable\n", name);
        return true;
    }
    const auto errors = Errors(mutated);
    std::printf("mutation %s: %zu errors%s\n", name, errors, errors ? "" : " (UNDETECTED)");
    return errors > 0;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <scenario.json>\n", argv[0]);
        return 2;
    }
    auto scenario = io::LoadScenario(argv[1]);
    engine::SolveProblem problem{game::CompileGame(scenario.game), std::move(scenario.ranges)};
    const engine::HandTraversalData data(problem, problem.game->Root());
    const Plan plan(data);
    std::size_t kinds[4] = {}, lane1 = 0, fork = 0, wait = 0, join = 0;
    for (const auto& p : plan.passes)
    {
        ++kinds[static_cast<int>(p.operation)];
        lane1 += p.lane;
        fork += (p.sync & kForkAfter) != 0;
        wait += (p.sync & kWaitFork) != 0;
        join += (p.sync & kJoinBefore) != 0;
    }
    std::printf(
        "%s: hands %u/%u stride %u, %zu passes (reach %zu terminal %zu backup %zu), lane-1 passes %zu, fork %zu wait %zu join %zu, "
        "terminal shared %zu B\n",
        argv[1],
        plan.state.hands[0],
        plan.state.hands[1],
        plan.state.stride,
        plan.passes.size(),
        kinds[0],
        kinds[1],
        kinds[2],
        lane1,
        fork,
        wait,
        join,
        TerminalSharedBytes(plan.state)
    );
    int failures = 0;
    for (U32 player = 0; player < 2; ++player)
    {
        const auto check = emu::CheckLanes(plan, player);
        std::printf("lane check player %u: %zu unordered pass pairs, %zu errors\n", player, check.unorderedPairs, check.errors.size());
        for (std::size_t i = 0; i < check.errors.size() && i < 5; ++i)
            std::printf("  %s\n", check.errors[i].c_str());
        failures += !check.errors.empty();
    }
    if (lane1 && !failures)
    {
        const auto clear = [](U32 bit)
        {
            return [bit](Plan& p)
            {
                bool changed = false;
                for (auto& pass : p.passes)
                    if (pass.sync & bit)
                    {
                        pass.sync &= ~bit;
                        changed = true;
                    }
                return changed;
            };
        };
        failures += !Detected(plan, "no-join", clear(kJoinBefore));
        failures += !Detected(plan, "no-fork-wait", clear(kWaitFork));
        // Point a lane-1 terminal's value slot at a lane-0 terminal's from the same forked region,
        // as an undersized lane stride would.
        failures += !Detected(
            plan,
            "overlapping-scratch",
            [](Plan& p)
            {
                std::size_t forked = p.passes.size();
                U32 lane0 = kNoIndex;
                for (std::size_t i = 0; i < p.passes.size(); ++i)
                {
                    const auto& pass = p.passes[i];
                    if (forked == p.passes.size() && (pass.sync & kForkAfter))
                        forked = i;
                    else if (i > forked && pass.operation == Kernel::Terminal && pass.count && !pass.lane && lane0 == kNoIndex)
                        lane0 = p.nodes[pass.offset].slot;
                    else if (i > forked && pass.operation == Kernel::Terminal && pass.count && pass.lane && lane0 != kNoIndex)
                    {
                        p.nodes[pass.offset].slot = lane0;
                        return true;
                    }
                }
                return false;
            }
        );
    }
    std::printf(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
