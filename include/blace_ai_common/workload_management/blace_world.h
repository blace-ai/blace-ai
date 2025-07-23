#pragma once

#include "library_defines.h"
#include <map>
#include <memory> // for shared_ptr
#include <mutex>
#include <string>
#include <vector> // for vector
namespace blace {
namespace workload_management {

class EXPORT_OR_IMPORT BlaceWorld {
public:
  BlaceWorld();
  ~BlaceWorld();
};

} // namespace workload_management
} // namespace blace