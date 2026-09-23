#include "engine/GpuDcfrSession.h"
#include <algorithm>
#include <stdexcept>

namespace solver::engine
{
GpuDcfrSession::GpuDcfrSession(std::shared_ptr<const SolveProblem> problem) : problem_(std::move(problem))
{
    if (!problem_ || !problem_->game)
        throw std::invalid_argument("GPU DCFR session requires a solve problem");
    if (!Available())
        throw std::runtime_error("A supported CUDA or Apple Silicon GPU is unavailable");
    const auto counts = MeasureSolveSize(*problem_);
    const auto& size = counts.tree;
    if (counts.strategyEntries > gpu::DeviceMemoryBudget() / (2 * sizeof(float)))
        throw std::runtime_error("Regret and cumulative strategy alone exceed GPU memory; reduce the tree or use CPU");
    data_ = std::make_shared<const HandTraversalData>(*problem_, problem_->game->Root());
    const gpu::Plan plan(*data_);
    state_ = plan.state;
    const auto fixed = HandTraversal::EstimateStorage(*problem_->game, counts.hands);
    // Combined host/device allocation estimate, not a claim about available VRAM.
    // Metal retains a copy of the update passes after the temporary upload plan dies.
    const auto host = size.storageBytes + fixed.fixedBytes + plan.passes.size() * sizeof(gpu::Pass);
    const auto sumsBytes = plan.Buffers()[gpu::SumsBuffer].bytes;
    const auto staging = std::min(sumsBytes, gpu::kReadbackBytes);
    const auto maxHands = std::max(counts.hands[0], counts.hands[1]);
    const auto evaluationVectors = sizeof(float) * (counts.hands[0] + counts.hands[1] + maxHands);
    const auto device = plan.DeviceBytes();
    const auto initialization = device + plan.HostBytes();
    const auto checkpointDownload = device + sumsBytes + staging;
    const auto certification = device + sumsBytes + fixed.workspaceBytes + evaluationVectors;
    const auto snapshot =
        StrategySnapshot::EstimateStorageBytes(size.decisionNodes[0] + size.decisionNodes[1], data_->infoSetCount, plan.entries);
    // Download releases other device buffers first; compaction reuses the host sums.
    const auto exportDownload = 2 * sumsBytes + staging;
    const auto exportSnapshot = snapshot + data_->maxActions * maxHands * sizeof(float);
    const auto peak = host + std::max<std::uint64_t>({initialization, checkpointDownload, certification, exportDownload, exportSnapshot});
    memory_ = {size.logicalNodes, size.topologyNodes, size.traversalNodes, counts.strategyEntries, peak + peak / 8 + 64 * 1024 * 1024, 0};
    executor_ = gpu::MakeExecutor(plan);
}

bool GpuDcfrSession::Available()
{
    return gpu::DeviceAvailable();
}

void GpuDcfrSession::Update(std::size_t player, float positiveDiscount, float averageDiscount)
{
    auto state = state_;
    state.player = static_cast<gpu::U32>(player);
    state.positiveDiscount = positiveDiscount;
    state.averageDiscount = averageDiscount;
    executor_->Update(state);
}

ExploitabilityMetrics GpuDcfrSession::EvaluateExploitability() const
{
    std::array<float, 2> response{};
    auto state = state_;
    state.evaluation = 1;
    for (gpu::U32 p = 0; p < 2; ++p)
    {
        state.player = p;
        const auto values = executor_->RootValues(state);
        float value = 0.0f, mass = 0.0f;
        for (std::size_t h = 0; h < values.size(); ++h)
        {
            const auto& hand = data_->tables->hands[p][h];
            const float weight = hand.weight * hand.opponentMass;
            value += weight * values[h];
            mass += weight;
        }
        response[p] = value / mass;
    }
    return {response[0], response[1], (response[0] + response[1]) / 2.0f};
}

ExploitabilityMetrics GpuDcfrSession::EvaluateExploitabilityOnCpu() const
{
    const auto sums = executor_->DownloadSums(false);
    return EvaluateAverageStrategy(HandTraversal(data_), sums.data());
}

void GpuDcfrSession::WriteTrainingState(const TrainingState& state)
{
    if (state.regrets.size() != data_->strategySize || state.strategySums.size() != data_->strategySize)
        throw std::invalid_argument("Training state does not match the GPU strategy layout");
    executor_->UploadTraining(state);
}

StrategySnapshot GpuDcfrSession::ExportStrategy() &&
{
    auto sums = executor_->DownloadSums(true);
    return data_->ExportStrategy(std::move(sums));
}
} // namespace solver::engine
