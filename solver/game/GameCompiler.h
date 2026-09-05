#pragma once

#include "game/GameSpec.h"
#include <memory>

namespace solver::game
{
class CompiledGame;

std::shared_ptr<const CompiledGame> CompileGame(const GameSpec& gameSpec);
} // namespace solver::game
