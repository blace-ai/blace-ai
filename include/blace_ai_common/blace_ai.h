#pragma once

#include "computation_graph/public_ops.h"
#include "util/util_common.h"
#include "workload_management/blace_world.h"

#ifndef BLACE_AI_USE_IPC
#include "computation_graph/graph_evaluator.h"
#else
#include "ipc_evaluator/ipc_evaluator.h"
#endif
