#pragma once

#include "ml_core/types.h"
#include <variant>
#include <vector>

#include "library_defines.h"

/**
 * Use this macro to get the constructed node to be used as input for other
 * nodes.
 */
#define CONSTRUCT_OP(op) blace::ops::create(op)

/**
 * @file public_ops.h
 * @brief This file contains all operators that can be used to build a
 * computation graph.
 */

namespace blace {
class NodeOp;
typedef std::shared_ptr<NodeOp> NodeOpP;
} // namespace blace

namespace blace {
namespace ops {
/**
 * Base class for all operators. Don't use directly.
 */
class BaseOp {

public:
  EXPORT_OR_IMPORT void set_hash(ml_core::BlaceHash new_hash);

  std::optional<std::shared_ptr<NodeOp>> getImpl() { return _impl; }

  EXPORT_OR_IMPORT int get_width();
  EXPORT_OR_IMPORT int get_height();
  EXPORT_OR_IMPORT void tag_for_caching(bool cache);
  EXPORT_OR_IMPORT ml_core::DeviceEnum get_device();

  /**
   * Internal storage of the constructed object.
   */
  std::optional<std::shared_ptr<NodeOp>> _impl;
};

/**
 * Returns the constructed object which can be used as input for later
 * operators.
 *
 * \return std::nullopt if construction failed, the constructed elemtent
 * otherwise.
 */
template <typename T> static std::shared_ptr<BaseOp> create(T node) {
  std::shared_ptr<BaseOp> x = std::make_shared<T>(node);

  return x;
}

typedef std::shared_ptr<BaseOp> OpP;
typedef std::vector<OpP> OpPVec;

} // namespace ops
} // namespace blace