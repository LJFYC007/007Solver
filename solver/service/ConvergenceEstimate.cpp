#include "service/ConvergenceEstimate.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace solver::service
{
void ConvergenceEstimate::Observe(int iterations, float exploitability, float trainingSeconds, float evaluationSeconds)
{
    if (size_ == samples_.size())
    {
        std::move(samples_.begin() + 1, samples_.end(), samples_.begin());
        --size_;
    }
    samples_[size_++] = {iterations, exploitability, trainingSeconds, evaluationSeconds};
}

std::optional<float> ConvergenceEstimate::TargetIteration(float target) const
{
    if (size_ == 0 || target <= 0.0f)
        return std::nullopt;
    const auto& latest = samples_[size_ - 1];
    if (latest.exploitability <= target)
        return static_cast<float>(latest.iterations);
    if (size_ < 3 || latest.exploitability >= samples_[size_ - 2].exploitability)
        return std::nullopt;

    float meanX = 0.0f, meanY = 0.0f;
    for (std::size_t i = 0; i < size_; ++i)
    {
        meanX += std::log(static_cast<float>(samples_[i].iterations));
        meanY += std::log(samples_[i].exploitability);
    }
    meanX /= size_;
    meanY /= size_;
    float xx = 0.0f, xy = 0.0f, yy = 0.0f;
    for (std::size_t i = 0; i < size_; ++i)
    {
        const float x = std::log(static_cast<float>(samples_[i].iterations)) - meanX;
        const float y = std::log(samples_[i].exploitability) - meanY;
        xx += x * x;
        xy += x * y;
        yy += y * y;
    }
    // A plateau or a noisy trend has no useful convergence-time prediction.
    if (!(xx > 0.0f && yy > 0.0f && xy < 0.0f) || xy * xy / (xx * yy) < 0.9f)
        return std::nullopt;
    const float rate = -xy / xx;
    const auto& previous = samples_[size_ - 2];
    const float recentRate =
        std::log(previous.exploitability / latest.exploitability) / std::log(static_cast<float>(latest.iterations) / previous.iterations);
    // A recent change in convergence speed invalidates an otherwise clean historical fit.
    if (recentRate < 0.5f * rate || recentRate > 2.0f * rate)
        return std::nullopt;
    const float predicted = latest.iterations * std::exp(std::log(latest.exploitability / target) / rate);
    return std::isfinite(predicted) ? std::optional<float>(predicted) : std::nullopt;
}

float ConvergenceEstimate::SecondsPerIteration() const
{
    const auto& latest = samples_[size_ - 1];
    const auto& first = samples_[0];
    return size_ == 1 ? latest.trainingSeconds / latest.iterations
                      : (latest.trainingSeconds - first.trainingSeconds) / (latest.iterations - first.iterations);
}

float ConvergenceEstimate::CheckInterval() const
{
    // Away from the target, spend roughly one eighth of training time on checks.
    return std::max(20.0f, 2.0f * std::ceil(8.0f * samples_[size_ - 1].evaluationSeconds / SecondsPerIteration() / 2.0f));
}

float ConvergenceEstimate::MinimumInterval() const
{
    return std::max(2.0f, 2.0f * std::ceil(samples_[size_ - 1].evaluationSeconds / SecondsPerIteration() / 2.0f));
}

int ConvergenceEstimate::NextCheck(int iterationLimit, float target) const
{
    if (size_ == 0)
        return std::min(20, iterationLimit);
    const auto& latest = samples_[size_ - 1];
    if (size_ < 3)
        return static_cast<int>(std::min<std::int64_t>(iterationLimit, 2LL * latest.iterations));

    float interval = CheckInterval();
    if (const auto predicted = TargetIteration(target))
    {
        // Check near the predicted crossing, without repeatedly paying for tiny batches.
        interval = std::max(MinimumInterval(), std::min(interval, 2.0f * std::ceil((*predicted - latest.iterations) / 2.0f)));
    }
    // Clamp before narrowing: float rounds INT_MAX up and loses unit steps above 2^24.
    const int remaining = iterationLimit - latest.iterations;
    if (!(interval < remaining))
        return iterationLimit;
    return latest.iterations + std::max(1, static_cast<int>(interval));
}

std::optional<float> ConvergenceEstimate::RemainingSeconds(int completed, int nextCheck, int iterationLimit, float target) const
{
    const auto predicted = TargetIteration(target);
    if (!predicted)
        return std::nullopt;
    const float distance = std::max(0.0f, std::min(static_cast<float>(iterationLimit), *predicted) - nextCheck);
    const float batches = std::floor(distance / CheckInterval());
    const float remainder = distance - batches * CheckInterval();
    const float tail = remainder > 0.0f ? std::max(MinimumInterval(), 2.0f * std::ceil(remainder / 2.0f)) : 0.0f;
    const float finish = std::min(static_cast<float>(iterationLimit), nextCheck + batches * CheckInterval() + tail);
    // Project onto the same check schedule, including the next in-flight evaluation.
    const float checks = 1.0f + batches + (tail > 0.0f ? 1.0f : 0.0f);
    return std::max(0.0f, finish - completed) * SecondsPerIteration() + checks * samples_[size_ - 1].evaluationSeconds +
           FinalizationSeconds();
}

float ConvergenceEstimate::FinalizationSeconds() const
{
    // Export and evaluation both traverse the policy; the latest check supplies its scale.
    return size_ == 0 ? 0.0f : samples_[size_ - 1].evaluationSeconds;
}
} // namespace solver::service
