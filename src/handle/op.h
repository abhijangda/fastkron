#include "fastkron.h"
#include "config.h"

#pragma once

std::string fastKronOpToStr(const fastKronOp& op);
std::ostream& operator<<(std::ostream& os, const fastKronOp& op);
fastKronOp swapFastKronOp(fastKronOp op);