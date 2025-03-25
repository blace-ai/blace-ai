#pragma once

#include "ml_core/callbacks.h"
#include <climits>
#include <memory>
#include <optional>
#include <stdint.h>
#include <vector>

namespace blace {
namespace ops {
class BaseOp;
typedef std::shared_ptr<blace::ops::BaseOp> OpP;
} // namespace ops
} // namespace blace

namespace blace {

namespace ml_core {

/**
 * Struct to describe a subset of a tensors dimension.
 */
struct Slice final {
public:
  /**
   * Construct a slice to select values along a dimension.
   * Slice() will select all values.
   * Slice(3,10) will select values in the range 3 to 10 (exclusive).
   *
   * \param start_index
   * \param stop_index
   * \param step_index
   */
  Slice(std::optional<int64_t> start_index = std::nullopt,
        std::optional<int64_t> stop_index = std::nullopt,
        std::optional<int64_t> step_index = std::nullopt);

  /**
   * Return the start value.
   *
   * \return
   */
  inline int64_t start() const { return start_; }

  /**
   * Return the end value.
   *
   * \return
   */
  inline int64_t stop() const { return stop_; }

  /**
   * Return the step value.
   *
   * \return
   */
  inline int64_t step() const { return step_; }

private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

enum class BlaceIndexType { None, Slice, OpP, Int };

/**
 * Struct to represent indices of a tensor along one dimension. This mirrors a
 * subset of https://pytorch.org/cppdocs/notes/tensor_indexing.html.
 */
struct BlaceIndex {
  /**
   *
   * Constructs an empty index.
   */
  BlaceIndex() : type_(BlaceIndexType::None) {}

  /**
   * Selects one entry along dimension.
   *
   * \param integer The entry to select.
   */
  BlaceIndex(int64_t integer) : integer_(integer), type_(BlaceIndexType::Int) {}

  /**
   * Slices a subset of a dimension.
   *
   * \param slice The slice.
   */
  BlaceIndex(blace::ml_core::Slice slice)
      : slice_(std::move(slice)), type_(BlaceIndexType::Slice) {}

  /**
   * Uses an node operator to select values from the input tensor. This is the
   * lazy evaluation equivalent of tensor[torch.tensor([1, 2])].
   *
   * \param tensor The node operator.
   */
  BlaceIndex(blace::ops::OpP tensor)
      : node_op_to_eval_(std::move(tensor)), type_(BlaceIndexType::OpP) {}

  /**
   * Check if type is none.
   *
   * \return
   */
  bool is_none() const;

  /**
   * Check if type is integer.
   *
   * \return
   */
  bool is_integer() const;

  /**
   * Return the stored integer.
   *
   * \return
   */
  int64_t integer() const;

  /**
   * Check if type is slice.
   *
   * \return
   */
  bool is_slice() const;

  /**
   * Return the slice.
   *
   * \return
   */
  const blace::ml_core::Slice &slice() const;

  /**
   * Check if type is node op.
   *
   * \return
   */
  bool is_node_op() const;

  /**
   * Return the operator.
   *
   * \return
   */
  const blace::ops::OpP &node_op() const;

  /**
   * Equality operator.
   *
   * \param rhs The other index.
   * \return
   */
  bool operator==(const BlaceIndex &rhs) const;

  /**
   * Get the type of the index.
   *
   * \return
   */
  BlaceIndexType getType();

  /**
   * Get the slice.
   *
   * \return
   */
  blace::ml_core::Slice getSlice();

  /**
   * Get the node operator.
   *
   * \return
   */
  blace::ops::OpP getNodeOp();

private:
  int64_t integer_ = 0;
  blace::ml_core::Slice slice_;
  blace::ops::OpP node_op_to_eval_;
  BlaceIndexType type_;
};

typedef std::vector<BlaceIndex> BlaceIndexVec;
} // namespace ml_core
} // namespace blace