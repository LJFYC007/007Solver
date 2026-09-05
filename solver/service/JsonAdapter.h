#pragma once

#include "service/ServiceMessage.h"
#include <string>

namespace solver::service
{
ServiceRequest ParseServiceRequest(const std::string& jsonLine);
std::string ServiceMessageToJson(const ServiceMessage& message);
} // namespace solver::service
