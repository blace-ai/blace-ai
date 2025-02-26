#pragma once

#include "ml_core/types.h" // for basic_types::ColorFormatMessage

namespace blace {

class RawMemoryMetadata {
public:
  RawMemoryMetadata();

  RawMemoryMetadata(ml_core::BlaceHash hash);

  ml_core::BlaceHash get_hash();

private:
  ml_core::BlaceHash _hash;
};

/*
Stores information about some image stored in memory. This should be filled
without accessing the actual memory itself.
*/
class RawImageMemoryMetadata : public RawMemoryMetadata {
public:
  RawImageMemoryMetadata(ml_core::BlaceHash hash, int width, int height,
                         ml_core::ColorFormatEnum color_format)
      : RawMemoryMetadata(hash) {
    _width = width;
    _height = height;
    _color_format = color_format;
  };

  int get_width();

  int get_height();

  ml_core::ColorFormatEnum get_color_format();

  static int channels_from_format(ml_core::ColorFormatEnum format);

private:
  int _width, _height;
  ml_core::ColorFormatEnum _color_format;
};

} // namespace blace