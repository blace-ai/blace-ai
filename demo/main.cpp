#ifdef BLACE_OS_MAC
#include <mach-o/dyld.h>
#elif BLACE_OS_WIN
#include <shlwapi.h>
#endif
//#include <unistd.h>

#include <string>
#include <variant>


#include "RunDepthModel.h"
#include <opencv2/opencv.hpp>

namespace blace {
namespace {
void program() { cv::Mat my_cv_mat = blace::runDepthModel(); }
} // namespace
} // namespace blace

int main(int argc, char *argv[]) {
  ::blace::program();
  return 0;
}
