#pragma once
#include "library_defines.h"
#include "ml_core/types.h"
#include <filesystem>
#include <vector>

/**
 * @file util.h
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
 * @brief Re.
 * @param model_message_file Path to a .blacemodel file which stores meta data
 * bytes.
 * @param model_folder The folder of the stored model.
 * @return The path of the executable.
 */

EXPORT_OR_IMPORT ml_core::ModelIdent
registerModel(std::string model_message_file,
              std::filesystem::path model_folder);

/**
 * @brief Re.
 * @param model_message_bytes Model bytes from header.
 * @param model_folder The folder of the stored model.
 * @return The path of the executable.
 */

EXPORT_OR_IMPORT ml_core::ModelIdent
registerModel(std::vector<char> model_message_bytes,
              std::filesystem::path model_folder);

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

} // namespace util
} // namespace blace
