#pragma once

#include "analysis/NodeReport.h"
#include "analysis/EquityReport.h"
#include "game/Identifiers.h"
#include "engine/MemoryEstimate.h"
#include <cstdint>
#include <optional>
#include <string>

namespace solver::service
{
enum class ServiceMessageKind : std::uint8_t
{
    BuildingTree,
    Solving,
    Ready,
    Failed,
    QuerySucceeded,
    QueryFailed,
};

enum class SolvePhase : std::uint8_t
{
    Training,
    Checking,
    Finalizing,
    Complete,
};

enum class StopReason : std::uint8_t
{
    Accuracy,
    IterationLimit,
};

struct SolveProgress
{
    SolvePhase phase = SolvePhase::Training;
    float elapsedSeconds = 0.0f;
    std::optional<float> estimatedRemainingSeconds;
    std::optional<float> accuracyPercent;
    float targetAccuracyPercent = 0.01f;
};

struct ServiceMessage
{
    ServiceMessageKind kind;
    std::uint64_t requestId = 0;
    int completedIterations = 0;
    int totalIterations = 0;
    int nodeCount = 0;
    game::NodeId rootNodeId{0};
    std::string text;
    std::optional<analysis::NodeReport> node;
    std::optional<analysis::EquityReport> equity;
    std::optional<engine::MemoryEstimate> estimate;
    SolveProgress progress;
    StopReason stopReason = StopReason::IterationLimit;
};

enum class QueryKind : std::uint8_t
{
    Node,
    Equity,
};

struct ServiceRequest
{
    std::uint64_t requestId = 0;
    std::optional<game::NodeId> nodeId;
    std::string validationError;
    QueryKind kind = QueryKind::Node;
};
} // namespace solver::service
