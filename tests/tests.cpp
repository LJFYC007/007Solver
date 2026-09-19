#include "analysis/AnalysisSession.h"
#include "engine/DcfrSession.h"
#include "engine/StrategyEvaluator.h"
#include "game/GameCompiler.h"
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
    auto scenario = solver::io::LoadScenario(std::string(TEST_FIXTURE_DIR) + name + ".json");
    return std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game::CompileGame(scenario.game), std::move(scenario.ranges)});
}

void CheckSolve(const std::string& name, engine::ComputeDevice requestedDevice, int workers)
{
    const auto problem = Problem(name);
    const int iterations = Fixture(name).at("iterations").get<int>();
    SCOPED_TRACE("dcfr workers=" + std::to_string(workers));
    engine::ExploitabilityMetrics checkpoint;
    // The service checks accuracy before consuming the training state.
    const auto strategy = [&]
    {
        engine::DcfrSession session(problem, requestedDevice, workers);
        if (requestedDevice == engine::ComputeDevice::Auto)
            EXPECT_EQ(session.Device(), engine::ComputeDevice::Gpu);
        testing::Test::RecordProperty("device", session.DeviceName());
        testing::Test::RecordProperty("cpuWorkers", std::to_string(session.WorkerCount()));
        std::cout << name << ": Device: " << session.DeviceName() << "; CPU workers: " << session.WorkerCount() << std::endl;
        // An odd split exercises player alternation across continued runs.
        session.Run(101);
        session.Run(iterations - 101);
        EXPECT_EQ(session.CompletedIterations(), iterations);
        checkpoint = session.EvaluateExploitability();
        return std::move(session).ExportStrategy();
    }();
    const auto actual = engine::EvaluateExploitability(*problem, strategy);
    EXPECT_NEAR(checkpoint.player0BestResponseEv, actual.player0BestResponseEv, 1e-6f);
    EXPECT_NEAR(checkpoint.player1BestResponseEv, actual.player1BestResponseEv, 1e-6f);
    const auto& expected = References().at(name).at("solved");
    ASSERT_TRUE(std::isfinite(actual.player0BestResponseEv));
    ASSERT_TRUE(std::isfinite(actual.player1BestResponseEv));
    ASSERT_TRUE(std::isfinite(actual.exploitability));
    // [-villain BR, hero BR] must intersect the external value interval.
    constexpr float roundingTolerance = 1e-5f;
    EXPECT_GE(actual.player0BestResponseEv, -expected.at("villainBestResponseEv").get<float>() - roundingTolerance);
    EXPECT_GE(actual.player1BestResponseEv, -expected.at("heroBestResponseEv").get<float>() - roundingTolerance);
    EXPECT_NEAR(actual.exploitability, (actual.player0BestResponseEv + actual.player1BestResponseEv) / 2.0f, roundingTolerance);
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

TEST(StrategyEvaluatorTest, UniformPolicyMatchesIndependentBestResponses)
{
    const auto problem = Problem("weighted-flop");
    const auto actual = engine::EvaluateExploitability(*problem, engine::StrategySnapshot(problem->game, {}));
    const auto& expected = References().at("weighted-flop").at("uniform");
    EXPECT_NEAR(actual.player0BestResponseEv, expected.at("heroBestResponseEv").get<float>(), 1e-5f);
    EXPECT_NEAR(actual.player1BestResponseEv, expected.at("villainBestResponseEv").get<float>(), 1e-5f);
    EXPECT_NEAR(actual.exploitability, expected.at("exploitability").get<float>(), 1e-5f);
}

TEST(SolverReferenceTest, WeightedFlopCpuOneWorker)
{
    CheckSolve("weighted-flop", engine::ComputeDevice::Cpu, 1);
}

TEST(SolverReferenceTest, WeightedFlopCpuFourWorkers)
{
    CheckSolve("weighted-flop", engine::ComputeDevice::Cpu, 4);
}

TEST(SolverReferenceTest, WeightedFlopGpu)
{
    if (!engine::GpuDcfrSession::Available())
        GTEST_SKIP() << "No supported CUDA or Metal GPU is available";
    CheckSolve("weighted-flop", engine::ComputeDevice::Auto, 0);
}

TEST(SolverReferenceTest, RaiseFlopCpuOneWorker)
{
    CheckSolve("raise-flop", engine::ComputeDevice::Cpu, 1);
}

TEST(SolverReferenceTest, RaiseFlopCpuFourWorkers)
{
    CheckSolve("raise-flop", engine::ComputeDevice::Cpu, 4);
}

TEST(SolverReferenceTest, RaiseFlopGpu)
{
    if (!engine::GpuDcfrSession::Available())
        GTEST_SKIP() << "No supported CUDA or Metal GPU is available";
    CheckSolve("raise-flop", engine::ComputeDevice::Auto, 0);
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
            EXPECT_FALSE(node.evsReady);
            node = session.EvaluateNodeEvs(std::move(node));
            EXPECT_TRUE(node.evsReady);
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
            }
        }
    }
}
