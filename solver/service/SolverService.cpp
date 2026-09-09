#include "service/SolverService.h"
#include "analysis/AnalysisSession.h"
#include "engine/CpuDcfrSession.h"
#include "engine/CpuEscfrSession.h"
#include "engine/StrategyEvaluator.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include "service/JsonAdapter.h"
#include "service/ServiceMessage.h"
#include <iostream>
#include <memory>
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
        io::Scenario scenario = io::LoadScenario(scenarioPath);
        WriteMessage(output_, {ServiceMessageKind::BuildingTree, 0, 0, scenario.iterations});

        diagnostics_ << "Start building decision tree...\n";
        std::shared_ptr<const game::CompiledGame> game = game::CompileGame(scenario.game);
        diagnostics_ << "Decision tree has " << game->NodeCount() << " nodes\n";

        auto problem = std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game, std::move(scenario.ranges)});

        diagnostics_ << "Start solving game...\n";
        WriteMessage(output_, {ServiceMessageKind::Solving, 0, 0, scenario.iterations});
        engine::SolveReport report{scenario.algorithm, "cpu"};
        const auto train = [&](auto& session)
        {
            session.Run(
                scenario.iterations,
                [&](int completedIterations)
                { WriteMessage(output_, {ServiceMessageKind::Solving, 0, completedIterations, scenario.iterations}); }
            );
            report.completedIterations = session.CompletedIterations();
            report.trainingTimeSeconds = session.TrainingTimeSeconds();
            return session.ExportStrategy();
        };
        engine::StrategySnapshot strategy = [&]
        {
            if (scenario.algorithm == "dcfr")
            {
                engine::CpuDcfrSession session(problem);
                auto snapshot = train(session);
                diagnostics_ << "Algorithm: dcfr, CPU workers: " << session.WorkerCount() << '\n';
                return snapshot;
            }
            engine::CpuEscfrSession session(problem);
            diagnostics_ << "Algorithm: escfr, CPU workers: 1\n";
            return train(session);
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
