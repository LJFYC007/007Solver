#pragma once

#include "core/Chips.h"
#include "engine/DcfrSession.h"
#include "engine/HandTraversalData.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

// Lockstep CPU/GPU checks (solver/ARCHITECTURE.md "Training"), shared by the correctness suite and the benchmark.
namespace solver::test
{
// Independent float evaluators agree within this many initial pots.
constexpr double kEvaluatorToleranceInPots = 5e-6;

inline void ExpectNodesNear(
    const engine::HandTraversalData& layout,
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const char* label
)
{
    ASSERT_EQ(expected.size(), layout.strategySize);
    ASSERT_EQ(actual.size(), expected.size());
    std::size_t mismatches = 0;
    for (const auto& node : layout.nodes)
    {
        if (node.kind != engine::HandTraversalData::Kind::Decision)
            continue;
        // Low-reach nodes hold tiny entries, so scale by each node's largest CPU entry.
        // Rounding stays below 1e-4 of that scale; update bugs move entries far more.
        const auto end = node.strategyOffset + node.childCount * layout.tables.hands[node.actor].size();
        float scale = 0.0f;
        for (auto i = node.strategyOffset; i < end; ++i)
            scale = std::max(scale, std::fabs(expected[i]));
        for (auto i = node.strategyOffset; i < end; ++i)
            if (!(std::fabs(actual[i] - expected[i]) <= 1e-3f * scale) && mismatches++ < 3)
                ADD_FAILURE() << label << " at node " << node.id.Value() << " entry " << i - node.strategyOffset << ": CPU " << expected[i]
                              << ", GPU " << actual[i] << ", node scale " << scale;
    }
    EXPECT_EQ(mismatches, 0u) << label;
}

inline std::shared_ptr<const engine::SolveProblem> MakeProblem(io::Scenario scenario)
{
    return std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game::CompileGame(scenario.game), std::move(scenario.ranges)});
}

// Compares the fixture's updates one at a time, then both devices' evaluations of the last state. Independent
// trajectories diverge at near-tied regrets, so every GPU update starts from the CPU state.
inline void CheckBackendParity(const std::string& fixture)
{
    auto scenario = io::LoadScenario(std::string(TEST_FIXTURE_DIR) + fixture + ".json");
    const int updates = scenario.iterations;
    const auto problem = MakeProblem(std::move(scenario));
    const engine::HandTraversalData layout(*problem, problem->game->Root());
    engine::DcfrSession cpu(problem, engine::ComputeDevice::Cpu);
    engine::DcfrSession gpu(problem, engine::ComputeDevice::Gpu);
    testing::Test::RecordProperty("device", gpu.DeviceName());
    auto expected = cpu.ReadTrainingState();
    for (int update = 0; update < updates; ++update)
    {
        SCOPED_TRACE("update " + std::to_string(update));
        gpu.WriteTrainingState(expected);
        cpu.Run(1);
        gpu.Run(1);
        expected = cpu.ReadTrainingState();
        const auto reference = layout.Decode(expected);
        const auto actual = layout.Decode(gpu.ReadTrainingState());
        ExpectNodesNear(layout, reference.regrets, actual.regrets, "regrets");
        ExpectNodesNear(layout, reference.strategySums, actual.strategySums, "strategy sums");
        EXPECT_EQ(actual.stamps, reference.stamps) << "node stamps";
        ASSERT_FALSE(testing::Test::HasFailure());
    }
    gpu.WriteTrainingState(expected);
    const auto reference = cpu.EvaluateCheckpoint();
    const auto actual = gpu.EvaluateCheckpoint();
    const double tolerance = kEvaluatorToleranceInPots * core::ToChipUnits(problem->game->Spec().initialPot);
    EXPECT_NEAR(actual.player0BestResponseEv, reference.player0BestResponseEv, tolerance);
    EXPECT_NEAR(actual.player1BestResponseEv, reference.player1BestResponseEv, tolerance);
    EXPECT_NEAR(actual.exploitability, reference.exploitability, tolerance);
}
} // namespace solver::test
