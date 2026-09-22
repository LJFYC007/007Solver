#include "service/SolverService.h"
#include "analysis/AnalysisSession.h"
#include "engine/DcfrSession.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include "service/ConvergenceEstimate.h"
#include "service/JsonAdapter.h"
#include "service/ServiceMessage.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace solver::service
{
namespace
{
void WriteMessage(std::ostream& output, const ServiceMessage& message)
{
    output << ServiceMessageToJson(message) << '\n';
    output.flush();
}
// EV traversal owns its scratch and only reads the immutable solve result.
// Navigation keeps the reach cache on the input thread and never waits for EVs.
class NodeEvWorker
{
public:
    NodeEvWorker(const analysis::AnalysisSession& analysis, std::ostream& output, std::mutex& outputMutex)
        : analysis_(analysis), output_(output), outputMutex_(outputMutex), thread_([this] { Run(); })
    {}

    ~NodeEvWorker()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        thread_.join();
    }

    void Enqueue(ServiceMessage response)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_.push_back(std::move(response));
        }
        ready_.notify_one();
    }

private:
    void Run()
    {
        for (;;)
        {
            ServiceMessage response{ServiceMessageKind::QuerySucceeded};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
                if (pending_.empty())
                    return;
                response = std::move(pending_.front());
                pending_.pop_front();
            }
            try
            {
                response.node = analysis_.EvaluateNodeEvs(std::move(*response.node));
            }
            catch (const std::exception& error)
            {
                response.kind = ServiceMessageKind::QueryFailed;
                response.text = error.what();
                response.node.reset();
            }
            std::lock_guard<std::mutex> lock(outputMutex_);
            WriteMessage(output_, response);
        }
    }

    const analysis::AnalysisSession& analysis_;
    std::ostream& output_;
    std::mutex& outputMutex_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<ServiceMessage> pending_;
    bool stopping_ = false;
    std::thread thread_;
};
} // namespace

SolverService::SolverService(std::istream& input, std::ostream& output, std::ostream& diagnostics)
    : input_(input), output_(output), diagnostics_(diagnostics)
{}

int SolverService::Run(const std::string& scenarioPath, engine::ComputeDevice device)
{
    try
    {
        io::Scenario scenario = [&]
        {
            if (scenarioPath != "--stdin")
                return io::LoadScenario(scenarioPath);
            std::string line;
            if (!std::getline(input_, line))
                throw std::invalid_argument("Expected one scenario JSON line on stdin");
            std::istringstream scenarioInput(line);
            return io::ReadScenario(scenarioInput);
        }();
        using Clock = std::chrono::steady_clock;
        const auto solveStart = Clock::now();
        const auto elapsed = [&] { return std::chrono::duration<float>(Clock::now() - solveStart).count(); };
        WriteMessage(output_, {ServiceMessageKind::BuildingTree, 0, 0, scenario.iterations});

        diagnostics_ << "Start building decision tree...\n";
        std::shared_ptr<const game::CompiledGame> game = game::CompileGame(scenario.game);
        diagnostics_ << "Decision tree has " << game->NodeCount() << " nodes\n";

        auto problem = std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game, std::move(scenario.ranges)});
        engine::MemoryEstimate estimate{};

        diagnostics_ << "Start solving game...\n";
        const float initialPot = static_cast<float>(game->Spec().initialPot.Raw()) / core::Chips::kUnitsPerChip;
        const float target = initialPot * (scenario.accuracyPercent / 100.0f);
        engine::ExploitabilityMetrics metrics;
        int completedIterations = 0;
        float trainingSeconds = 0.0f;
        SolveProgress solveProgress;
        solveProgress.targetAccuracyPercent = scenario.accuracyPercent;
        engine::StrategySnapshot strategy = [&]
        {
            engine::DcfrSession session(problem, device);
            estimate = session.Memory();
            diagnostics_ << "Device: " << session.DeviceName() << "; CPU workers: " << session.WorkerCount() << "; "
                         << estimate.strategyEntries << " strategy entries; combined allocation peak estimate " << estimate.peakBytes
                         << " bytes\n";
            ConvergenceEstimate convergence;
            int nextCheck = convergence.NextCheck(scenario.iterations, target);
            const auto progress = [&](int completed, SolvePhase phase)
            {
                ServiceMessage message{ServiceMessageKind::Solving, 0, completed, scenario.iterations};
                message.estimate = estimate;
                solveProgress.phase = phase;
                solveProgress.elapsedSeconds = elapsed();
                solveProgress.estimatedRemainingSeconds =
                    phase == SolvePhase::Finalizing ? convergence.FinalizationSeconds()
                                                    : convergence.RemainingSeconds(completed, nextCheck, scenario.iterations, target);
                message.progress = solveProgress;
                WriteMessage(output_, message);
            };
            progress(0, SolvePhase::Training);
            while (session.CompletedIterations() < scenario.iterations)
            {
                session.Run(nextCheck - session.CompletedIterations(), [&](int completed) { progress(completed, SolvePhase::Training); });
                const int completed = session.CompletedIterations();
                progress(completed, SolvePhase::Checking);
                const auto checkStart = Clock::now();
                metrics = session.EvaluateCheckpoint(completed == scenario.iterations, target);
                const float evaluationSeconds = std::chrono::duration<float>(Clock::now() - checkStart).count();
                convergence.Observe(completed, metrics.exploitability, session.TrainingTimeSeconds(), evaluationSeconds);
                if (initialPot > 0.0f)
                    solveProgress.accuracyPercent = 100.0f * std::max(0.0f, metrics.exploitability) / initialPot;
                if (metrics.exploitability <= target)
                    break;

                nextCheck = convergence.NextCheck(scenario.iterations, target);
            }
            completedIterations = session.CompletedIterations();
            trainingSeconds = session.TrainingTimeSeconds();
            progress(completedIterations, SolvePhase::Finalizing);
            return std::move(session).ExportStrategy();
        }();
        diagnostics_ << "Training time: " << trainingSeconds << " s\n";

        diagnostics_ << "Iteration: " << completedIterations << " Hero BR EV: " << metrics.player0BestResponseEv
                     << " Villain BR EV: " << metrics.player1BestResponseEv << " Exploitability: " << metrics.exploitability << '\n';

        const bool targetReached = metrics.exploitability <= target;
        analysis::AnalysisSession analysis(engine::SolveResult(problem, std::move(strategy)));
        diagnostics_ << "Solving complete.\n";
        ServiceMessage ready{ServiceMessageKind::Ready};
        ready.completedIterations = completedIterations;
        ready.nodeCount = static_cast<int>(game->NodeCount());
        ready.rootNodeId = analysis.RootNode();
        ready.estimate = estimate;
        solveProgress.phase = SolvePhase::Complete;
        solveProgress.elapsedSeconds = elapsed();
        solveProgress.estimatedRemainingSeconds = 0.0f;
        ready.progress = solveProgress;
        ready.stopReason = targetReached ? StopReason::Accuracy : StopReason::IterationLimit;
        WriteMessage(output_, ready);

        std::mutex outputMutex;
        NodeEvWorker evWorker(analysis, output_, outputMutex);
        std::string requestLine;
        while (std::getline(input_, requestLine))
        {
            const ServiceRequest request = ParseServiceRequest(requestLine);
            try
            {
                if (!request.validationError.empty())
                    throw std::invalid_argument(request.validationError);
                ServiceMessage response{ServiceMessageKind::QuerySucceeded};
                response.requestId = request.requestId;
                if (request.kind == QueryKind::Equity)
                    response.equity = analysis.QueryEquity(*request.nodeId);
                else
                    response.node = analysis.QueryNode(*request.nodeId);
                if (request.kind == QueryKind::NodeEvs)
                    evWorker.Enqueue(std::move(response));
                else
                {
                    std::lock_guard<std::mutex> lock(outputMutex);
                    WriteMessage(output_, response);
                }
            }
            catch (const std::exception& error)
            {
                ServiceMessage response{ServiceMessageKind::QueryFailed};
                response.requestId = request.requestId;
                response.text = error.what();
                std::lock_guard<std::mutex> lock(outputMutex);
                WriteMessage(output_, response);
            }
        }
    }
    catch (const std::exception& error)
    {
        return Fail(error.what());
    }
    return 0;
}

int SolverService::Fail(const std::string& message)
{
    ServiceMessage failed{ServiceMessageKind::Failed};
    failed.text = message;
    WriteMessage(output_, failed);
    return 1;
}
} // namespace solver::service
