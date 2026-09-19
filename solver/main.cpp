#include "service/SolverService.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    solver::service::SolverService service(std::cin, std::cout, std::cerr);
    if (argc != 2 && argc != 3)
        return service.Fail("Expected a scenario path or --stdin, optionally followed by --device=cpu|gpu|auto");
    auto device = solver::engine::ComputeDevice::Auto;
    if (argc == 3)
    {
        const std::string option = argv[2];
        if (option == "--device=cpu")
            device = solver::engine::ComputeDevice::Cpu;
        else if (option == "--device=gpu")
            device = solver::engine::ComputeDevice::Gpu;
        else if (option != "--device=auto")
            return service.Fail("Expected --device=cpu|gpu|auto");
    }
    return service.Run(argv[1], device);
}
