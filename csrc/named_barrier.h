#pragma once

#include "cutlass/barrier.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
    SReady = 1,
    SoftmaxReady = 2,  // 用于同步Softmax计算完成状态
};

} // flash
