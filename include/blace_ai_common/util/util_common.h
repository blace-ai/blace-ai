#pragma once
#include "library_defines.h"
#include "ml_core/types.h"
#include <filesystem>
#include <vector>

/**
 * @file util_common.h
 * @brief This file contains some util functions and classes.
 */

namespace blace {
namespace util {
/**
 * @brief Get the path to the running executable.
 * @return The path of the executable.
 */

EXPORT_OR_IMPORT std::filesystem::path getPathToExe();

/**
 * @brief Gets the available accelerator. CPU, CUDA or MPS.
 * @return The available accelerator or nullopt if accelerator should be present
 * but returned error.
 */
EXPORT_OR_IMPORT std::optional<blace::ml_core::DeviceEnum> get_accelerator();

} // namespace util
} // namespace blace
