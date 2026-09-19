#pragma once

#include "engine/DcfrSession.h"
#include <iosfwd>
#include <string>

namespace solver::service
{
class SolverService
{
public:
    SolverService(std::istream& input, std::ostream& output, std::ostream& diagnostics);

    int Run(const std::string& scenarioPath, engine::ComputeDevice device = engine::ComputeDevice::Auto);
    int Fail(const std::string& message);

private:
    std::istream& input_;
    std::ostream& output_;
    std::ostream& diagnostics_;
};
} // namespace solver::service
