#include "BackendParity.h"
#include "analysis/AnalysisSession.h"
#include "engine/DcfrSession.h"
#include "engine/StrategyEvaluator.h"
#include "io/ScenarioLoader.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace
{
namespace analysis = solver::analysis;
namespace core = solver::core;
namespace engine = solver::engine;
namespace game = solver::game;
using Json = nlohmann::json;

Json Fixture(const std::string& name)
{
    return Json::parse(std::ifstream(std::string(TEST_FIXTURE_DIR) + name + ".json"));
}

const Json& References()
{
    static const Json references = Fixture("correctness-reference");
    return references;
}

std::shared_ptr<const engine::SolveProblem> Problem(const std::string& name)
{
    // Changing the input requires regenerating its independent reference.
    EXPECT_EQ(Fixture(name), References().at(name).at("scenario"));
    return solver::test::MakeProblem(solver::io::LoadScenario(std::string(TEST_FIXTURE_DIR) + name + ".json"));
}

void CheckSolve(const std::string& name, engine::ComputeDevice requestedDevice)
{
    const auto problem = Problem(name);
    const int iterations = Fixture(name).at("iterations").get<int>();
    const auto& reference = References().at(name);
    engine::ExploitabilityMetrics checkpoint;
    // The service checks accuracy before consuming the training state.
    const auto strategy = [&]
    {
        // Several CPU workers split chance subtrees; the GPU ignores the count.
        engine::DcfrSession session(problem, requestedDevice, 4);
        if (requestedDevice == engine::ComputeDevice::Auto)
            EXPECT_EQ(session.Device(), engine::ComputeDevice::Gpu);
        testing::Test::RecordProperty("device", session.DeviceName());
        testing::Test::RecordProperty("cpuWorkers", std::to_string(session.WorkerCount()));
        std::cout << name << ": Device: " << session.DeviceName() << "; CPU workers: " << session.WorkerCount() << std::endl;
        if (reference.contains("uniform"))
        {
            // Before any update, the zero cumulative strategies normalize to uniform play.
            const auto uniform = session.EvaluateCheckpoint();
            const auto& expected = reference.at("uniform");
            EXPECT_NEAR(uniform.player0BestResponseEv, expected.at("heroBestResponseEv").get<float>(), 1e-5f);
            EXPECT_NEAR(uniform.player1BestResponseEv, expected.at("villainBestResponseEv").get<float>(), 1e-5f);
            EXPECT_NEAR(uniform.exploitability, expected.at("exploitability").get<float>(), 1e-5f);
        }
        // An odd split exercises player alternation across continued runs.
        session.Run(101);
        session.Run(iterations - 101);
        EXPECT_EQ(session.CompletedIterations(), iterations);
        checkpoint = session.EvaluateCheckpoint();
        return std::move(session).ExportStrategy();
    }();
    // CPU evaluation of the export cross-checks the training device's evaluation.
    const auto actual = engine::EvaluateExploitability(*problem, strategy);
    EXPECT_NEAR(checkpoint.player0BestResponseEv, actual.player0BestResponseEv, 1e-6f);
    EXPECT_NEAR(checkpoint.player1BestResponseEv, actual.player1BestResponseEv, 1e-6f);
    // Raked checkpoints also depend on each player's average-strategy value.
    EXPECT_NEAR(checkpoint.exploitability, actual.exploitability, 1e-6f);
    const auto& expected = reference.at("solved");
    ASSERT_TRUE(std::isfinite(actual.player0BestResponseEv));
    ASSERT_TRUE(std::isfinite(actual.player1BestResponseEv));
    ASSERT_TRUE(std::isfinite(actual.exploitability));
    // Zero-sum games only: [-villain BR, hero BR] must intersect the external value interval.
    // Raked references are checked by the uniform and fixed-policy evaluations instead.
    if (!problem->game->Spec().HasRake())
    {
        constexpr float roundingTolerance = 1e-5f;
        EXPECT_GE(actual.player0BestResponseEv, -expected.at("villainBestResponseEv").get<float>() - roundingTolerance);
        EXPECT_GE(actual.player1BestResponseEv, -expected.at("heroBestResponseEv").get<float>() - roundingTolerance);
        EXPECT_NEAR(actual.exploitability, (actual.player0BestResponseEv + actual.player1BestResponseEv) / 2.0f, roundingTolerance);
    }
    EXPECT_LE(actual.exploitability, 0.01f); // 0.5% of the fixtures' initial pot.
    ASSERT_FALSE(testing::Test::HasFailure());
}

engine::StrategySnapshot FixedStrategy(const engine::SolveProblem& problem, const Json& policy)
{
    std::vector<engine::StrategyEntry> entries;
    for (const auto& node : policy)
    {
        auto nodeId = problem.game->Root();
        for (const auto& action : node.at("path"))
            nodeId = problem.game->GetNode(nodeId).GetBettingEdge(action.get<std::size_t>()).NextNode();
        for (const auto& hand : node.at("hands"))
        {
            entries.push_back({
                {nodeId, core::ParseHoleCards(hand.at("cards").get<std::string>())},
                hand.at("strategy").get<std::vector<float>>(),
            });
        }
    }
    return engine::StrategySnapshot(problem.game, std::move(entries));
}
} // namespace

TEST(SolverReferenceTest, WeightedFlopCpu)
{
    CheckSolve("weighted-flop", engine::ComputeDevice::Cpu);
}

TEST(SolverReferenceTest, WeightedFlopGpu)
{
    if (!engine::GpuDcfrSession::Available())
        GTEST_SKIP() << "No supported CUDA GPU is available";
    CheckSolve("weighted-flop", engine::ComputeDevice::Auto);
}

TEST(SolverReferenceTest, RaiseFlopCpu)
{
    CheckSolve("raise-flop", engine::ComputeDevice::Cpu);
}

TEST(SolverReferenceTest, RaiseFlopGpu)
{
    if (!engine::GpuDcfrSession::Available())
        GTEST_SKIP() << "No supported CUDA GPU is available";
    CheckSolve("raise-flop", engine::ComputeDevice::Auto);
}

TEST(BackendParityTest, GpuUpdatesMatchCpuFromSharedState)
{
    if (!engine::GpuDcfrSession::Available())
        GTEST_SKIP() << "No supported CUDA GPU is available";
    solver::test::CheckBackendParity("backend-parity");
}

TEST(AnalysisSessionTest, FixedPoliciesMatchIndependentNodeValuesAndReach)
{
    for (const std::string name : {"weighted-flop", "raise-flop"})
    {
        const auto problem = Problem(name);
        const auto& reference = References().at(name);
        analysis::AnalysisSession session(engine::SolveResult(problem, FixedStrategy(*problem, reference.at("policy"))));
        for (const auto& expected : reference.at("queries"))
        {
            SCOPED_TRACE(name + " path " + expected.at("path").dump());
            auto node = session.QueryNode(session.RootNode());
            for (const auto& step : expected.at("path"))
            {
                if (step.is_number())
                {
                    node = session.QueryNode(node.actions.at(step.get<std::size_t>()).nextNodeId);
                }
                else
                {
                    const auto card = core::ParseCard(step.get<std::string>());
                    const auto outcome = std::find_if(
                        node.outcomes.begin(), node.outcomes.end(), [&](const auto& candidate) { return candidate.card == card; }
                    );
                    ASSERT_NE(outcome, node.outcomes.end());
                    node = session.QueryNode(outcome->nextNodeId);
                }
            }
            ASSERT_EQ(node.kind, game::NodeKind::Decision);
            EXPECT_EQ(node.actor, core::PlayerId(expected.at("actor").get<std::uint8_t>()));
            EXPECT_EQ(node.state.board, core::ParseBoard(expected.at("board").get<std::string>(), core::BoardCardCount(node.state.street)));
            EXPECT_DOUBLE_EQ(static_cast<double>(node.state.pot.Raw()) / core::Chips::kUnitsPerChip, expected.at("pot").get<double>());
            for (std::size_t player = 0; player < 2; ++player)
                EXPECT_DOUBLE_EQ(
                    static_cast<double>(node.state.stacks[player].Raw()) / core::Chips::kUnitsPerChip,
                    expected.at("stacks").at(player).get<double>()
                );
            ASSERT_EQ(node.hands.size(), expected.at("hands").size());
            for (const auto& hand : expected.at("hands"))
            {
                SCOPED_TRACE(hand.at("cards").get<std::string>());
                const auto cards = core::ParseHoleCards(hand.at("cards").get<std::string>());
                const auto actual =
                    std::find_if(node.hands.begin(), node.hands.end(), [&](const auto& candidate) { return candidate.cards == cards; });
                ASSERT_NE(actual, node.hands.end());
                EXPECT_FLOAT_EQ(actual->inputRangeWeight, hand.at("inputRangeWeight").get<float>());
                EXPECT_FLOAT_EQ(actual->ownReachWeight, hand.at("ownReachWeight").get<float>());
                const float mass = hand.at("marginalReachMass").get<float>();
                EXPECT_NEAR(actual->marginalReachMass, mass, std::max(1e-12f, mass * 1e-5f));
                ASSERT_EQ(actual->nodeStrategyEv.has_value(), !hand.at("nodeStrategyEv").is_null());
                if (actual->nodeStrategyEv.has_value())
                    EXPECT_NEAR(*actual->nodeStrategyEv, hand.at("nodeStrategyEv").get<float>(), 1e-5f);
                EXPECT_EQ(actual->strategy, hand.at("strategy").get<std::vector<float>>());
                const auto actionEvs = hand.at("actionEvs").get<std::vector<float>>();
                ASSERT_EQ(actual->actionEvs.size(), actionEvs.size());
                for (std::size_t action = 0; action < actionEvs.size(); ++action)
                    EXPECT_NEAR(actual->actionEvs[action], actionEvs[action], 1e-5f);
            }
            const auto opponent = session.QueryOpponentEv(node.nodeId);
            EXPECT_EQ(opponent.player, node.actor->Other());
            ASSERT_EQ(opponent.hands.size(), expected.at("opponentHands").size());
            for (const auto& hand : expected.at("opponentHands"))
            {
                SCOPED_TRACE("opponent " + hand.at("cards").get<std::string>());
                const auto cards = core::ParseHoleCards(hand.at("cards").get<std::string>());
                const auto actual = std::find_if(
                    opponent.hands.begin(), opponent.hands.end(), [&](const auto& candidate) { return candidate.cards == cards; }
                );
                ASSERT_NE(actual, opponent.hands.end());
                const float mass = hand.at("marginalReachMass").get<float>();
                EXPECT_NEAR(actual->marginalReachMass, mass, std::max(1e-12f, mass * 1e-5f));
                ASSERT_EQ(actual->nodeStrategyEv.has_value(), !hand.at("nodeStrategyEv").is_null());
                if (actual->nodeStrategyEv.has_value())
                    EXPECT_NEAR(*actual->nodeStrategyEv, hand.at("nodeStrategyEv").get<float>(), 1e-5f);
            }
        }
    }
}
