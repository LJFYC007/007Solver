// BackendParityTest's lockstep comparison for any scenario: each GPU update starts from
// the CPU state; entries must match within 1e-3 of each node's largest CPU entry, stamps exactly.
// Usage: daily_parity <scenario.json> <updates> [cpu-workers]
#include "engine/DcfrSession.h"
#include "engine/HandTraversalData.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
using namespace solver;
std::size_t Mismatches(
    const engine::HandTraversalData& layout,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float& worst
)
{
    std::size_t mismatches = 0;
    for (const auto& node : layout.nodes)
    {
        if (node.kind != engine::HandTraversalData::Kind::Decision)
            continue;
        const auto end = node.strategyOffset + node.childCount * layout.tables->hands[node.actor].size();
        float scale = 0.0f;
        for (auto i = node.strategyOffset; i < end; ++i)
            scale = std::max(scale, std::fabs(expected[i]));
        for (auto i = node.strategyOffset; i < end; ++i)
        {
            const float error = std::fabs(actual[i] - expected[i]);
            if (scale > 0)
                worst = std::max(worst, error / scale);
            if (!(error <= 1e-3f * scale))
                ++mismatches;
        }
    }
    return mismatches;
}
int main(int argc, char** argv)
{
    auto scenario = io::LoadScenario(argv[1]);
    auto problem =
        std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game::CompileGame(scenario.game), std::move(scenario.ranges)});
    const engine::HandTraversalData layout(*problem, problem->game->Root());
    engine::DcfrSession cpu(problem, engine::ComputeDevice::Cpu, argc > 3 ? std::atoi(argv[3]) : 4);
    engine::DcfrSession gpu(problem, engine::ComputeDevice::Gpu);
    auto expected = cpu.ReadTrainingState();
    int failures = 0;
    for (int update = 0; update < std::atoi(argv[2]); ++update)
    {
        gpu.WriteTrainingState(expected);
        cpu.Run(1);
        gpu.Run(1);
        expected = cpu.ReadTrainingState();
        const auto actual = gpu.ReadTrainingState();
        float worstRegret = 0, worstSum = 0;
        const auto r = Mismatches(layout, expected.regrets, actual.regrets, worstRegret);
        const auto s = Mismatches(layout, expected.strategySums, actual.strategySums, worstSum);
        const bool stamps = actual.stamps == expected.stamps;
        std::printf(
            "%s update %d: regret mismatches %zu (worst %.2e of node scale), sum mismatches %zu (worst %.2e), stamps %s\n",
            gpu.DeviceName(),
            update,
            r,
            worstRegret,
            s,
            worstSum,
            stamps ? "equal" : "DIFFER"
        );
        std::fflush(stdout);
        failures += r || s || !stamps;
    }
    return failures ? 1 : 0;
}
