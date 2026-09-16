#include "service/SolverService.h"
#include "analysis/AnalysisSession.h"
#include "engine/CpuDcfrSession.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include "service/ConvergenceEstimate.h"
#include "service/JsonAdapter.h"
#include "service/ServiceMessage.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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
} // namespace

SolverService::SolverService(std::istream& input, std::ostream& output, std::ostream& diagnostics)
    : input_(input), output_(output), diagnostics_(diagnostics)
{}

int SolverService::Run(const std::string& scenarioPath)
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
        const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - solveStart).count(); };
        WriteMessage(output_, {ServiceMessageKind::BuildingTree, 0, 0, scenario.iterations});

        diagnostics_ << "Start building decision tree...\n";
        std::shared_ptr<const game::CompiledGame> game = game::CompileGame(scenario.game);
        diagnostics_ << "Decision tree has " << game->NodeCount() << " nodes\n";

        auto problem = std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game, std::move(scenario.ranges)});
        const auto estimate = engine::EstimateCpuMemory(*problem);
        diagnostics_ << "Tree estimate: " << estimate.logicalNodes << " logical nodes, " << estimate.topologyNodes << " topology nodes, "
                     << estimate.traversalNodes << " active nodes, " << estimate.strategyEntries
                     << " strategy entries; solve peak estimate " << estimate.peakBytes << " bytes\n";

        diagnostics_ << "Start solving game...\n";
        const double initialPot = static_cast<double>(game->Spec().initialPot.Raw()) / core::Chips::kUnitsPerChip;
        const double target = initialPot * (scenario.accuracyPercent / 100.0);
        engine::ExploitabilityMetrics metrics;
        int completedIterations = 0;
        double trainingSeconds = 0.0;
        SolveProgress solveProgress;
        solveProgress.targetAccuracyPercent = scenario.accuracyPercent;
        engine::StrategySnapshot strategy = [&]
        {
            engine::CpuDcfrSession session(problem);
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
                metrics = session.EvaluateExploitability();
                const double evaluationSeconds = std::chrono::duration<double>(Clock::now() - checkStart).count();
                convergence.Observe(completed, metrics.exploitability, session.TrainingTimeSeconds(), evaluationSeconds);
                if (initialPot > 0.0)
                    solveProgress.accuracyPercent = 100.0 * std::max(0.0f, metrics.exploitability) / initialPot;
                if (metrics.exploitability <= target)
                    break;

                nextCheck = convergence.NextCheck(scenario.iterations, target);
            }
            completedIterations = session.CompletedIterations();
            trainingSeconds = session.TrainingTimeSeconds();
            diagnostics_ << "Algorithm: dcfr, configured CPU worker limit: " << session.WorkerCount() << '\n';
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
        solveProgress.estimatedRemainingSeconds = 0.0;
        ready.progress = solveProgress;
        ready.stopReason = targetReached ? StopReason::Accuracy : StopReason::IterationLimit;
        WriteMessage(output_, ready);

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
                if (request.equity)
                    response.equity = analysis.QueryEquity(*request.nodeId);
                else
                    response.node = analysis.QueryNode(*request.nodeId);
                WriteMessage(output_, response);
            }
            catch (const std::exception& error)
            {
                ServiceMessage response{ServiceMessageKind::QueryFailed};
                response.requestId = request.requestId;
                response.text = error.what();
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
