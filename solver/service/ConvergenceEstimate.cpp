#include "service/ConvergenceEstimate.h"
#include <algorithm>
#include <cmath>

namespace solver::service
{
void ConvergenceEstimate::Observe(int iterations, double exploitability, double trainingSeconds, double evaluationSeconds)
{
    if (size_ == samples_.size())
    {
        std::move(samples_.begin() + 1, samples_.end(), samples_.begin());
        --size_;
    }
    samples_[size_++] = {iterations, exploitability, trainingSeconds, evaluationSeconds};
}

std::optional<double> ConvergenceEstimate::TargetIteration(double target) const
{
    if (size_ == 0 || target <= 0.0)
        return std::nullopt;
    const auto& latest = samples_[size_ - 1];
    if (latest.exploitability <= target)
        return latest.iterations;
    if (size_ < 3 || latest.exploitability >= samples_[size_ - 2].exploitability)
        return std::nullopt;

    double meanX = 0.0, meanY = 0.0;
    for (std::size_t i = 0; i < size_; ++i)
    {
        meanX += std::log(samples_[i].iterations);
        meanY += std::log(samples_[i].exploitability);
    }
    meanX /= size_;
    meanY /= size_;
    double xx = 0.0, xy = 0.0, yy = 0.0;
    for (std::size_t i = 0; i < size_; ++i)
    {
        const double x = std::log(samples_[i].iterations) - meanX;
        const double y = std::log(samples_[i].exploitability) - meanY;
        xx += x * x;
        xy += x * y;
        yy += y * y;
    }
    // A plateau or a noisy trend has no useful convergence-time prediction.
    if (!(xx > 0.0 && yy > 0.0 && xy < 0.0) || xy * xy / (xx * yy) < 0.9)
        return std::nullopt;
    const double rate = -xy / xx;
    const auto& previous = samples_[size_ - 2];
    const double recentRate =
        std::log(previous.exploitability / latest.exploitability) / std::log(static_cast<double>(latest.iterations) / previous.iterations);
    // A recent change in convergence speed invalidates an otherwise clean historical fit.
    if (recentRate < 0.5 * rate || recentRate > 2.0 * rate)
        return std::nullopt;
    const double predicted = latest.iterations * std::exp(std::log(latest.exploitability / target) / rate);
    return std::isfinite(predicted) ? std::optional<double>(predicted) : std::nullopt;
}

double ConvergenceEstimate::SecondsPerIteration() const
{
    const auto& latest = samples_[size_ - 1];
    const auto& first = samples_[0];
    return size_ == 1 ? latest.trainingSeconds / latest.iterations
                      : (latest.trainingSeconds - first.trainingSeconds) / (latest.iterations - first.iterations);
}

double ConvergenceEstimate::CheckInterval() const
{
    // Away from the target, spend roughly one eighth of training time on checks.
    return std::max(20.0, 2.0 * std::ceil(8.0 * samples_[size_ - 1].evaluationSeconds / SecondsPerIteration() / 2.0));
}

double ConvergenceEstimate::MinimumInterval() const
{
    return std::max(2.0, 2.0 * std::ceil(samples_[size_ - 1].evaluationSeconds / SecondsPerIteration() / 2.0));
}

int ConvergenceEstimate::NextCheck(int iterationLimit, double target) const
{
    if (size_ == 0)
        return std::min(20, iterationLimit);
    const auto& latest = samples_[size_ - 1];
    if (size_ < 3)
        return static_cast<int>(std::min(static_cast<double>(iterationLimit), 2.0 * latest.iterations));

    double interval = CheckInterval();
    if (const auto predicted = TargetIteration(target))
    {
        // Check near the predicted crossing, without repeatedly paying for tiny batches.
        interval = std::max(MinimumInterval(), std::min(interval, 2.0 * std::ceil((*predicted - latest.iterations) / 2.0)));
    }
    return static_cast<int>(std::min(static_cast<double>(iterationLimit), latest.iterations + interval));
}

std::optional<double> ConvergenceEstimate::RemainingSeconds(int completed, int nextCheck, int iterationLimit, double target) const
{
    const auto predicted = TargetIteration(target);
    if (!predicted)
        return std::nullopt;
    const double distance = std::max(0.0, std::min(static_cast<double>(iterationLimit), *predicted) - nextCheck);
    const double batches = std::floor(distance / CheckInterval());
    const double remainder = distance - batches * CheckInterval();
    const double tail = remainder > 0.0 ? std::max(MinimumInterval(), 2.0 * std::ceil(remainder / 2.0)) : 0.0;
    const double finish = std::min(static_cast<double>(iterationLimit), nextCheck + batches * CheckInterval() + tail);
    // Project onto the same check schedule, including the next in-flight evaluation.
    const double checks = 1.0 + batches + (tail > 0.0 ? 1.0 : 0.0);
    return std::max(0.0, finish - completed) * SecondsPerIteration() + checks * samples_[size_ - 1].evaluationSeconds +
           FinalizationSeconds();
}

double ConvergenceEstimate::FinalizationSeconds() const
{
    // Export and evaluation both traverse the policy; the latest check supplies its scale.
    return size_ == 0 ? 0.0 : samples_[size_ - 1].evaluationSeconds;
}
} // namespace solver::service
