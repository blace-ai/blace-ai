#ifdef BLACE_OS_MAC
#include <mach-o/dyld.h>
#elif BLACE_OS_WIN
#include <shlwapi.h>
#endif
//#include <unistd.h>

#include <string>
#include <variant>

#include "RunDepthModel.h"

namespace blace {
namespace {
void program() { blace::run(); }
} // namespace
} // namespace blace

int main(int argc, char *argv[]) {
  ::blace::program();
  return 0;
}
