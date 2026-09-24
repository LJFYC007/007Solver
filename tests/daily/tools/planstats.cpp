#include "engine/HandTraversalData.h"
#include "engine/gpu/GpuPlan.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include <cstdio>
using namespace solver;
using namespace solver::engine::gpu;
int main(int argc, char** argv)
{
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
}
