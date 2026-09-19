#include "engine/GpuDcfrSession.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace solver::engine
{
GpuDcfrSession::GpuDcfrSession(std::shared_ptr<const SolveProblem> problem) : problem_(std::move(problem))
{
    if (!problem_ || !problem_->game)
        throw std::invalid_argument("GPU DCFR session requires a solve problem");
    if (!Available())
        throw std::runtime_error("A supported CUDA or Apple Silicon GPU is unavailable");
    memory_ = EstimateCpuMemory(*problem_, 1);
    if (memory_.strategyEntries > gpu::DeviceMemoryBudget() / (2 * sizeof(float)))
        throw std::runtime_error("Regret and cumulative strategy alone exceed GPU memory; reduce the tree or use CPU");
    data_ = std::make_shared<const HandTraversalData>(*problem_, problem_->game->Root());
    const gpu::Plan plan(*data_);
    state_ = plan.state;
    memory_.workers = 0;
    const auto fixed = HandTraversal::EstimateStorage(*problem_->game, {data_->hands[0].size(), data_->hands[1].size()});
    // Combined host/device allocation estimate, not a claim about available VRAM.
    const auto host = problem_->game->Size().storageBytes + fixed.fixedBytes;
    const auto scratch = plan.slots * (state_.hands[0] + state_.hands[1] + 3 * state_.stride) * sizeof(float);
    const auto compilation = plan.DeviceBytes() - 2 * plan.entries * sizeof(float) - scratch - plan.outcomeEntries * sizeof(gpu::U32) -
                             sizeof(gpu::State) + plan.passes.size() * sizeof(gpu::Pass);
    const auto certification = plan.entries * sizeof(float) + fixed.workspaceBytes;
    const auto size = problem_->game->Size();
    const auto snapshot =
        StrategySnapshot::EstimateStorageBytes(size.decisionNodes[0] + size.decisionNodes[1], data_->infoSetCount, plan.entries);
    const auto exportPeak = host + plan.entries * sizeof(float) + snapshot + 16 * 1024 * 1024;
    const auto peak = std::max(host + plan.DeviceBytes() + std::max(compilation, certification), exportPeak);
    memory_.peakBytes = peak + peak / 8 + 64 * 1024 * 1024;
    executor_ = gpu::MakeExecutor(plan);
}

bool GpuDcfrSession::Available()
{
    return gpu::DeviceAvailable();
}

void GpuDcfrSession::Run(int iterations, const std::function<void(int)>& progressCallback)
{
    if (exported_)
        throw std::logic_error("GPU training state has been exported");
    if (iterations <= 0 || iterations > std::numeric_limits<int>::max() - completedIterations_)
        throw std::invalid_argument("DCFR iterations must be positive and fit the completed iteration counter");
    const auto start = std::chrono::steady_clock::now();
    auto lastProgress = start;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        state_.player = completedIterations_ % 2;
        state_.evaluation = 0;
        const float t = completedIterations_ / 2 + 1.0f;
        const float power = t * std::sqrt(t);
        state_.positiveDiscount = power / (power + 1.0f);
        state_.averageDiscount = (t / (t + 1.0f)) * (t / (t + 1.0f));
        executor_->Update(state_);
        ++completedIterations_;
        if (progressCallback)
        {
            const auto now = std::chrono::steady_clock::now();
            if (iteration + 1 == iterations || now - lastProgress >= std::chrono::milliseconds(250))
            {
                progressCallback(completedIterations_);
                lastProgress = now;
            }
        }
    }
    trainingTimeSeconds_ += std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
}

ExploitabilityMetrics GpuDcfrSession::EvaluateExploitability() const
{
    if (exported_)
        throw std::logic_error("GPU training state has been exported");
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
            const auto& hand = data_->hands[p][h];
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
    if (exported_)
        throw std::logic_error("GPU training state has been exported");
    const auto sums = executor_->DownloadSums(false);
    return EvaluateAverageStrategy(HandTraversal(data_), sums.data());
}

StrategySnapshot GpuDcfrSession::ExportStrategy() &&
{
    if (exported_)
        throw std::logic_error("GPU training state has been exported");
    auto sums = executor_->DownloadSums(true);
    exported_ = true;
    return data_->ExportStrategy(std::move(sums));
}
} // namespace solver::engine
