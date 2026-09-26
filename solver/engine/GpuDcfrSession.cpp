#include "engine/GpuDcfrSession.h"
#include <algorithm>
#include <stdexcept>

namespace solver::engine
{
GpuDcfrSession::GpuDcfrSession(const SolveProblem& problem)
{
    if (!Available())
        throw std::runtime_error("A supported CUDA GPU is unavailable");
    const auto counts = MeasureSolveSize(problem);
    const auto& size = counts.tree;
    const auto budget = gpu::DeviceMemoryBudget();
    if (TrainingStateBytes(counts) > budget)
        throw std::runtime_error("Regret and cumulative strategy alone exceed GPU memory; reduce the tree or use CPU");
    data_ = std::make_shared<const HandTraversalData>(problem, problem.game->Root());
    const gpu::Plan plan(*data_);
    const auto device = plan.DeviceBytes();
    if (device > budget)
        throw std::runtime_error("GPU training state and batch scratch exceed available GPU memory");
    state_ = plan.state;
    const auto fixed = HandTraversal::EstimateStorage(*problem.game, counts.hands);
    // Combined host/device allocation estimate, not a claim about available VRAM.
    const auto host = size.storageBytes + fixed.fixedBytes;
    const auto sumsBytes = plan.Buffers()[gpu::SumsBuffer].bytes;
    const auto initialization = device + plan.HostBytes();
    // CPU certification evaluates a host copy of the sums while training stays resident.
    const auto certification = device + sumsBytes + fixed.WalkBytes(CpuWorkerCount());
    const auto snapshot = StrategySnapshot::EstimateStorageBytes(size.decisionNodes[0] + size.decisionNodes[1], counts.strategyEntries);
    // Download releases other device buffers first; the snapshot's probabilities are
    // normalized from the downloaded sums.
    const auto exportDownload = 2 * sumsBytes;
    const auto exportSnapshot = sumsBytes + snapshot;
    const auto peak = host + std::max<std::uint64_t>({initialization, certification, exportDownload, exportSnapshot});
    memory_ = MakeMemoryEstimate(counts, peak, 0);
    executor_ = gpu::MakeExecutor(plan);
}

bool GpuDcfrSession::Available()
{
    return gpu::DeviceAvailable();
}

void GpuDcfrSession::Update(std::size_t player, const UpdateWeights& weights)
{
    auto state = state_;
    state.player = static_cast<gpu::U32>(player);
    state.update = weights.update;
    state.positiveScale = weights.positiveScale;
    state.positiveInverse = weights.positiveInverse;
    state.averageWeight = weights.averageWeight;
    executor_->Update(state);
}

ExploitabilityMetrics GpuDcfrSession::EvaluateExploitability() const
{
    auto state = state_;
    state.evaluation = 1;
    std::array<std::vector<float>, 2> values;
    for (gpu::U32 p = 0; p < 2; ++p)
    {
        state.player = p;
        values[p] = executor_->RootValues(state);
    }
    return RootExploitability(data_->tables, values);
}

ExploitabilityMetrics GpuDcfrSession::EvaluateExploitabilityOnCpu() const
{
    const auto sums = executor_->DownloadSums(false);
    return EvaluateAverageStrategy(HandTraversal(data_), sums.data());
}

void GpuDcfrSession::WriteTrainingState(const QuantizedState& state)
{
    if (state.regrets.size() != data_->strategySize || state.strategySums.size() != data_->strategySize ||
        state.stamps.size() != data_->nodes.size())
        throw std::invalid_argument("Training state does not match the GPU strategy layout");
    executor_->UploadTraining(state);
}

StrategySnapshot GpuDcfrSession::ExportStrategy() &&
{
    return data_->ExportStrategy(executor_->DownloadSums(true));
}
} // namespace solver::engine
