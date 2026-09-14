#include "service/SolverService.h"
#include "analysis/AnalysisSession.h"
#include "engine/CpuDcfrSession.h"
#include "engine/StrategyEvaluator.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include "service/JsonAdapter.h"
#include "service/ServiceMessage.h"
#include "service/MemoryBudget.h"
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
        WriteMessage(output_, {ServiceMessageKind::BuildingTree, 0, 0, scenario.iterations});

        diagnostics_ << "Start building decision tree...\n";
        std::shared_ptr<const game::CompiledGame> game = game::CompileGame(scenario.game);
        diagnostics_ << "Decision tree has " << game->NodeCount() << " nodes\n";

        auto problem = std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game, std::move(scenario.ranges)});
        const auto estimate = engine::EstimateCpuMemory(*problem);
        const auto budget = scenario.memoryBudgetBytes ? scenario.memoryBudgetBytes : AvailableSolveMemory();
        diagnostics_ << "Tree estimate: " << estimate.logicalNodes << " logical nodes, " << estimate.topologyNodes << " topology nodes, "
                     << estimate.traversalNodes << " active nodes, " << estimate.strategyEntries
                     << " strategy entries; solve peak estimate " << estimate.peakBytes << " bytes; budget " << budget << " bytes\n";
        if (estimate.peakBytes > budget)
        {
            std::ostringstream message;
            message << "Estimated solve memory " << (estimate.peakBytes + 1048575) / 1048576 << " MiB exceeds the " << budget / 1048576
                    << " MiB memory budget. Free memory or choose a smaller game. "
                    << "CLI scenarios can set memoryBudgetMiB explicitly.";
            throw std::runtime_error(message.str());
        }

        diagnostics_ << "Start solving game...\n";
        const auto progress = [&](int completed)
        {
            ServiceMessage message{ServiceMessageKind::Solving, 0, completed, scenario.iterations};
            message.estimate = estimate;
            WriteMessage(output_, message);
        };
        progress(0);
        engine::SolveReport report{"dcfr", "cpu"};
        engine::StrategySnapshot strategy = [&]
        {
            engine::CpuDcfrSession session(problem);
            session.Run(scenario.iterations, progress);
            report.completedIterations = session.CompletedIterations();
            report.trainingTimeSeconds = session.TrainingTimeSeconds();
            diagnostics_ << "Algorithm: dcfr, configured CPU worker limit: " << session.WorkerCount() << '\n';
            return session.ExportStrategy();
        }();
        diagnostics_ << "Training time: " << report.trainingTimeSeconds << " s\n";

        diagnostics_ << "Evaluate exploitability...\n";
        report.metrics = engine::EvaluateExploitability(*problem, strategy);
        diagnostics_ << "Iteration: " << report.completedIterations << " Hero BR EV: " << report.metrics.player0BestResponseEv
                     << " Villain BR EV: " << report.metrics.player1BestResponseEv << " Exploitability: " << report.metrics.exploitability
                     << '\n';

        analysis::AnalysisSession analysis(engine::SolveResult(problem, std::move(strategy), std::move(report)));
        diagnostics_ << "Solving complete.\n";
        ServiceMessage ready{ServiceMessageKind::Ready};
        ready.completedIterations = scenario.iterations;
        ready.nodeCount = static_cast<int>(game->NodeCount());
        ready.rootNodeId = analysis.RootNode();
        ready.estimate = estimate;
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
