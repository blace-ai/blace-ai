#pragma once

#include "library_defines.h"
#include <map>
#include <memory> // for shared_ptr
#include <mutex>
#include <string>
#include <vector> // for vector

namespace blace {
namespace ipc_engine {
namespace utils {
class Manager;
}
} // namespace ipc_engine
} // namespace blace

namespace blace {
namespace workload_management {

class EXPORT_OR_IMPORT BlaceWorld {
public:
  BlaceWorld();
  ~BlaceWorld();
  static bool is_ipc_initialized();
  static ::blace::ipc_engine::utils::Manager *get_ipc_manager();
};

} // namespace workload_management
} // namespace blace