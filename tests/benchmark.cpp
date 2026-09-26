#include "analysis/AnalysisSession.h"
#include "engine/DcfrSession.h"
#include "engine/StrategyEvaluator.h"
#include "engine/MemoryEstimate.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include "service/JsonAdapter.h"
#if defined(_WIN32)
#include <Windows.h>
#include <Psapi.h>
#else
#include <mach/mach.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace
{
using namespace solver;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr double kTolerance = 1e-5;
// Compare independent float evaluators in initial-pot units.
constexpr double kReferenceToleranceInPots = 2e-6;
Json report;
std::filesystem::path reportPath;
Clock::time_point stageStart;
int iterationBudget = 0;
int workers = 0;
engine::ComputeDevice computeDevice = engine::ComputeDevice::Cpu;
bool checkConvergence = false;
bool stopAtAccuracy = false;

int PositiveInteger(const std::string& value)
{
    std::size_t parsed = 0;
    const int number = std::stoi(value, &parsed);
    if (number <= 0 || parsed != value.size())
        throw std::invalid_argument("Benchmark counts must be positive integers");
    return number;
}

void SaveReport()
{
    std::ofstream stream(reportPath);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << report.dump(2) << '\n';
}

Json Memory()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX memory{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory)))
        throw std::runtime_error("GetProcessMemoryInfo failed");
    return {{"private_committed_bytes", memory.PrivateUsage}, {"peak_working_set_bytes", memory.PeakWorkingSetSize}};
#else
    task_vm_info_data_t memory{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    rusage usage{};
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&memory), &count) != KERN_SUCCESS ||
        getrusage(RUSAGE_SELF, &usage) != 0)
        throw std::runtime_error("Cannot read macOS process memory");
    // These macOS metrics are not equivalent to Windows private committed memory.
    return {{"physical_footprint_bytes", memory.phys_footprint}, {"peak_resident_set_bytes", usage.ru_maxrss}};
#endif
}

void StartStage(const char* name)
{
    report["active_stage"] = name;
    SaveReport(); // Preserve the last completed phase even if a later phase is interrupted.
    std::cout << "Starting " << name << std::endl;
    stageStart = Clock::now();
}

void FinishStage(const char* name)
{
    const double seconds = std::chrono::duration<double>(Clock::now() - stageStart).count();
    report["stages"][name] = Memory();
    report["stages"][name]["seconds"] = seconds;
    report["active_stage"] = nullptr;
    SaveReport();
    std::cout << name << ": " << seconds << " s" << std::endl;
}

Json Metrics(const engine::ExploitabilityMetrics& metrics)
{
    return {
        {"heroBestResponseEv", metrics.player0BestResponseEv},
        {"villainBestResponseEv", metrics.player1BestResponseEv},
        {"exploitability", metrics.exploitability}
    };
}

void CheckMetrics(const engine::ExploitabilityMetrics& metrics)
{
    EXPECT_TRUE(std::isfinite(metrics.player0BestResponseEv));
    EXPECT_TRUE(std::isfinite(metrics.player1BestResponseEv));
    EXPECT_TRUE(std::isfinite(metrics.exploitability));
    EXPECT_GE(metrics.exploitability, -kTolerance);
    EXPECT_NEAR(metrics.exploitability, (metrics.player0BestResponseEv + metrics.player1BestResponseEv) / 2.0, kTolerance);
}

class Diagnostics : public testing::EmptyTestEventListener
{
    void OnTestPartResult(const testing::TestPartResult& result) override
    {
        if (result.failed())
        {
            report["failures"].push_back(result.message());
            SaveReport();
        }
    }
};
} // namespace

TEST(WideRangeBenchmark, UtgBbSingleRaisedFixedWork)
{
    StartStage("preparation");
    const std::string fixturePath = std::string(TEST_FIXTURE_DIR) + "utg-bb-wide.json";
    const Json input = Json::parse(std::ifstream(fixturePath));
    report["scenario"] = input;
    const Json reference = Json::parse(std::ifstream(std::string(TEST_FIXTURE_DIR) + "benchmark-reference.json"));
    report["oracle"] = reference.at("source");
    const auto& expected = reference.at("utg-bb-wide");
    ASSERT_EQ(input, expected.at("scenario"));
    ASSERT_EQ(input.at("initialPot"), 5.5);
    ASSERT_EQ(input.at("heroPosition"), "UTG");
    ASSERT_EQ(input.at("villainPosition"), "BB");
    ASSERT_EQ(reference.at("source").at("chipScale"), 10.0);
    ASSERT_EQ(reference.at("source").at("revision"), "9d1509fe5077d019825f833eed04b16d342dfda1");
    for (const auto* key : {"heroBestResponseEv", "villainBestResponseEv", "exploitability"})
    {
        ASSERT_TRUE(std::isfinite(expected.at("uniform").at(key).get<double>()));
    }
    auto scenario = io::LoadScenario(fixturePath);
    ASSERT_EQ(input.at("heroStack"), 97.5);
    ASSERT_EQ(input.at("villainStack"), 97.5);
    const double initialPot = input.at("initialPot").get<double>();
    const double targetExploitability = initialPot * scenario.accuracyPercent / 100.0;
    report["exploitability_limit"] = targetExploitability;
    const auto problem =
        std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game::CompileGame(scenario.game), std::move(scenario.ranges)});
    const int iterations = iterationBudget == 0 ? scenario.iterations : iterationBudget;
    report["iterations"] = iterations;
    report["node_count"] = problem->game->NodeCount();
    const auto estimate = engine::EstimateCpuMemory(*problem, workers);
    report["tree_estimate"] = {
        {"logical_nodes", estimate.logicalNodes},
        {"topology_nodes", estimate.topologyNodes},
        {"traversal_nodes", estimate.traversalNodes},
        {"strategy_entries", estimate.strategyEntries},
        {"peak_bytes", estimate.peakBytes},
        {"workers", estimate.workers}
    };
    std::array<std::size_t, 2> hands{};
    std::size_t pairs = 0;
    const auto& board = problem->game->Spec().initialBoard;
    for (std::uint8_t player = 0; player < 2; ++player)
        for (const auto& [hand, weight] : problem->ranges.For(core::PlayerId(player)).Entries())
            if (weight > 0.0f && !core::Overlaps(hand, board))
                ++hands[player];
    for (const auto& [hero, heroWeight] : problem->ranges.For(core::PlayerId::Player0()).Entries())
        for (const auto& [villain, villainWeight] : problem->ranges.For(core::PlayerId::Player1()).Entries())
            if (heroWeight > 0.0f && villainWeight > 0.0f && !core::Overlaps(hero, board) && !core::Overlaps(villain, board) &&
                !core::Overlaps(hero, villain))
                ++pairs;
    report["legal_hands"] = hands;
    report["legal_hand_pairs"] = pairs;
    FinishStage("preparation");

    StartStage("uniform_evaluation");
    const auto uniform = engine::EvaluateExploitability(*problem, engine::StrategySnapshot(problem->game, {}));
    report["uniform"] = Metrics(uniform);
    FinishStage("uniform_evaluation");
    CheckMetrics(uniform);
    EXPECT_NEAR(
        uniform.player0BestResponseEv / initialPot,
        expected.at("uniform").at("heroBestResponseEv").get<double>() / initialPot,
        kReferenceToleranceInPots
    );
    EXPECT_NEAR(
        uniform.player1BestResponseEv / initialPot,
        expected.at("uniform").at("villainBestResponseEv").get<double>() / initialPot,
        kReferenceToleranceInPots
    );
    EXPECT_NEAR(
        uniform.exploitability / initialPot,
        expected.at("uniform").at("exploitability").get<double>() / initialPot,
        kReferenceToleranceInPots
    );
    ASSERT_FALSE(HasFailure());

    StartStage("session_initialization");
    auto strategy = [&]
    {
        auto session = std::make_unique<engine::DcfrSession>(problem, computeDevice, workers);
        report["workers"] = session->WorkerCount();
        report["device"] = session->DeviceName();
        std::cout << "Device: " << session->DeviceName() << std::endl;
        report["tree_estimate"]["peak_bytes"] = session->Memory().peakBytes;
        report["tree_estimate"]["workers"] = session->WorkerCount();
        FinishStage("session_initialization");
        StartStage("training");
        int lastProgress = 0;
        const auto trainingStart = Clock::now();
        while (session->CompletedIterations() < iterations)
        {
            const int remaining = iterations - session->CompletedIterations();
            session->Run(
                checkConvergence ? (std::min)(200, remaining) : remaining,
                [&](int completed)
                {
                    if (completed - lastProgress >= 100 || completed == iterations)
                    {
                        std::cout << "Training: " << completed << " / " << iterations << " updates" << std::endl;
                        lastProgress = completed;
                    }
                }
            );
            if (checkConvergence)
            {
                const auto metrics = session->EvaluateCheckpoint(
                    session->CompletedIterations() == iterations,
                    stopAtAccuracy ? std::optional<double>(targetExploitability) : std::nullopt
                );
                CheckMetrics(metrics);
                report["checkpoint"] = Metrics(metrics);
                report["convergence"].push_back({
                    {"iterations", session->CompletedIterations()},
                    {"elapsed_seconds", std::chrono::duration<double>(Clock::now() - trainingStart).count()},
                    {"training_seconds", session->TrainingTimeSeconds()},
                    {"exploitability", metrics.exploitability},
                });
                SaveReport();
                if (stopAtAccuracy && metrics.exploitability <= targetExploitability)
                    break;
            }
        }
        report["training_loop_seconds"] = session->TrainingTimeSeconds();
        report["updates_per_second"] = session->CompletedIterations() / session->TrainingTimeSeconds();
        report["completed_iterations"] = session->CompletedIterations();
        FinishStage("training");
        if (stopAtAccuracy)
            EXPECT_LE(session->CompletedIterations(), iterations);
        else
            EXPECT_EQ(session->CompletedIterations(), iterations);
        StartStage("snapshot_export");
        auto snapshot = std::move(*session).ExportStrategy();
        FinishStage("snapshot_export");
        StartStage("training_release");
        session.reset();
        FinishStage("training_release");
        return snapshot;
    }();

    StartStage("trained_evaluation");
    const auto actual = engine::EvaluateExploitability(*problem, strategy);
    report["trained"] = Metrics(actual);
    report["trained"]["accuracyPercent"] = 100.0 * actual.exploitability / initialPot;
    report["target_reached"] = actual.exploitability <= targetExploitability;
    FinishStage("trained_evaluation");
    CheckMetrics(actual);
    if (checkConvergence)
        EXPECT_NEAR(actual.exploitability, report.at("checkpoint").at("exploitability").get<double>(), 1e-6);
    // Low iteration budgets measure throughput; convergence precision belongs to the correctness suite.
    EXPECT_LT(actual.exploitability, uniform.exploitability - kTolerance);
    ASSERT_FALSE(HasFailure());

    StartStage("analysis_initialization");
    analysis::AnalysisSession analysis(engine::SolveResult(problem, std::move(strategy)));
    FinishStage("analysis_initialization");
    StartStage("root_query");
    auto root = analysis.QueryNode(analysis.RootNode());
    FinishStage("root_query");
    ASSERT_EQ(root.kind, game::NodeKind::Decision);
    ASSERT_EQ(root.actor, core::PlayerId::Player1());
    EXPECT_EQ(root.hands.size(), hands[1]);
    EXPECT_EQ(root.state.board, board);
    EXPECT_EQ(root.state.pot, problem->game->Spec().initialPot);
    EXPECT_EQ(root.state.stacks, problem->game->Spec().initialStacks);
    EXPECT_FALSE(root.actions.empty());
    double mass = 0.0;
    for (const auto& hand : root.hands)
    {
        EXPECT_TRUE(std::isfinite(hand.inputRangeWeight));
        EXPECT_TRUE(std::isfinite(hand.ownReachWeight));
        EXPECT_TRUE(std::isfinite(hand.marginalReachMass));
        EXPECT_GT(hand.inputRangeWeight, 0.0f);
        EXPECT_EQ(hand.ownReachWeight, hand.inputRangeWeight);
        EXPECT_GT(hand.marginalReachMass, 0.0f);
        EXPECT_FALSE(core::Overlaps(hand.cards, board));
        EXPECT_TRUE(hand.nodeStrategyEv.has_value());
        if (hand.nodeStrategyEv)
            EXPECT_TRUE(std::isfinite(*hand.nodeStrategyEv));
        EXPECT_EQ(hand.strategy.size(), root.actions.size());
        for (const float probability : hand.strategy)
        {
            EXPECT_TRUE(std::isfinite(probability));
            EXPECT_GE(probability, 0.0f);
            EXPECT_LE(probability, 1.0f);
        }
        EXPECT_NEAR(std::accumulate(hand.strategy.begin(), hand.strategy.end(), 0.0), 1.0, kTolerance);
        mass += hand.marginalReachMass;
    }
    EXPECT_NEAR(mass, 1.0, kTolerance);
    // Reuse the service serializer so the report includes every current root-query field.
    service::ServiceMessage message{service::ServiceMessageKind::QuerySucceeded};
    message.node = std::move(root);
    report["root"] = Json::parse(service::ServiceMessageToJson(message)).at("node");
    SaveReport();
}

int main(int argc, char** argv)
{
    const auto start = Clock::now();
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#if defined(_WIN32)
    const auto processId = GetCurrentProcessId();
#else
    const auto processId = getpid();
#endif
    reportPath = "build/benchmark-results/utg-bb-wide-" + std::to_string(stamp) + "-" + std::to_string(processId) + ".json";
    bool reportCreated = false;
    try
    {
        for (int i = 1; i < argc;)
        {
            const std::string arg = argv[i];
            if (arg.rfind("--report=", 0) == 0)
                reportPath = arg.substr(9);
            else if (arg.rfind("--iterations=", 0) == 0)
                iterationBudget = PositiveInteger(arg.substr(13));
            else if (arg.rfind("--workers=", 0) == 0)
                workers = PositiveInteger(arg.substr(10));
            else if (arg == "--device=gpu")
                computeDevice = engine::ComputeDevice::Gpu;
            else if (arg == "--device=cpu")
                computeDevice = engine::ComputeDevice::Cpu;
            else if (arg == "--device=auto")
                computeDevice = engine::ComputeDevice::Auto;
            else if (arg == "--convergence")
                checkConvergence = true;
            else if (arg == "--stop-at-accuracy")
                checkConvergence = stopAtAccuracy = true;
            else
            {
                ++i;
                continue;
            }
            for (int j = i; j + 1 < argc; ++j)
                argv[j] = argv[j + 1];
            --argc;
            argv[argc] = nullptr;
        }
        if (reportPath.has_parent_path())
            std::filesystem::create_directories(reportPath.parent_path());
        if (std::filesystem::exists(reportPath))
            throw std::runtime_error("Report already exists; choose a new path: " + reportPath.string());
        report = {
            {"scenario_id", "utg-bb-wide"},
            {"algorithm", "dcfr"},
            {"iteration_unit", "full_player_update"},
            {"accuracy_stopping", stopAtAccuracy},
            {"status", "running"},
            {"failures", Json::array()},
            {"build",
             {{"compiler", BENCHMARK_COMPILER},
              {"configuration", BENCHMARK_CONFIGURATION},
              {"compiled_at", __DATE__ " " __TIME__},
              {"pointer_bits", sizeof(void*) * 8}}}
        };
        SaveReport();
        reportCreated = true;
        testing::InitGoogleTest(&argc, argv);
        testing::UnitTest::GetInstance()->listeners().Append(new Diagnostics);
        int result = RUN_ALL_TESTS();
        if (testing::UnitTest::GetInstance()->successful_test_count() != 1)
            result = 1;
        report["status"] = result == 0 ? "passed" : "failed";
        report["total_seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
        report["final_memory"] = Memory();
        SaveReport();
        std::cout << "Report: " << std::filesystem::absolute(reportPath).string() << std::endl;
        return result;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        if (reportCreated)
        {
            report["status"] = "failed";
            report["failures"].push_back(error.what());
            report["total_seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
            try
            {
                SaveReport();
            }
            catch (const std::exception& writeError)
            {
                std::cerr << "Cannot save diagnostic report: " << writeError.what() << std::endl;
            }
        }
        return 1;
    }
}
