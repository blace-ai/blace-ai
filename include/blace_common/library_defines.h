#pragma once

#ifdef BLACE_OS_WIN

#ifdef BLACE_LIBRARY_EXPORTED
#define EXPORT_OR_IMPORT __declspec(dllexport)
#else
#define EXPORT_OR_IMPORT __declspec(dllimport)
#endif

#elif defined(BLACE_OS_MAC) || defined(BLACE_OS_UBUNTU)
#ifdef BLACE_LIBRARY_EXPORTED
#define EXPORT_OR_IMPORT __attribute__((visibility("default")))
#else
#define EXPORT_OR_IMPORT
#endif

#else
#define EXPORT_OR_IMPORT

#endif
