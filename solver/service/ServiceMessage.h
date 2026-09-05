#pragma once

#include "analysis/NodeReport.h"
#include "game/Identifiers.h"
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
};

struct ServiceRequest
{
    std::uint64_t requestId = 0;
    std::optional<game::NodeId> nodeId;
    std::string validationError;
};
} // namespace solver::service
