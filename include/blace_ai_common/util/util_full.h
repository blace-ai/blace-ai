#pragma once
#include "computation_graph/public_base_op.h"
#include "library_defines.h"
#include "ml_core/types.h"
#include "util/util_common.h"
#include <filesystem>
#include <vector>

/**
 * @file util.h
 * @brief This file contains some util functions and classes.
 */

namespace blace {
namespace util {

/**
 * @brief Gets the available accelerator. CPU, CUDA or MPS.
 * @return The available accelerator or nullopt if accelerator should be present
 * but returned error.
 */
EXPORT_OR_IMPORT std::optional<blace::ml_core::DeviceEnum> get_accelerator();

/**
 * @brief Unload all loaded models.
 */
EXPORT_OR_IMPORT void unloadModels();

/**
 * @brief Checks if an operator is cached on server side. \param op The op.
 * @return True if the op is cached..
 */
EXPORT_OR_IMPORT bool
has_cached_value(std::shared_ptr<::blace::ops::BaseOp> op);

} // namespace util
} // namespace blace
