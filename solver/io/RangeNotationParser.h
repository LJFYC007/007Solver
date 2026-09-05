#pragma once

#include "core/Range.h"
#include <string>
#include <utility>
#include <vector>

namespace solver::io
{
core::Range ParseRangeNotation(const std::vector<std::pair<std::string, float>>& weightedHandClasses);
} // namespace solver::io
