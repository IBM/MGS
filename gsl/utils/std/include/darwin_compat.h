#ifndef DARWIN_COMPAT_H
#define DARWIN_COMPAT_H

#ifdef __APPLE__
// Include macOS equivalents
#include <limits.h>
#include <float.h>

// Define Linux values.h macros using macOS equivalents
#define MININT INT_MIN
#define MAXINT INT_MAX
#define MINFLOAT FLT_MIN
#define MAXFLOAT FLT_MAX
#define MINDOUBLE DBL_MIN
#define MAXDOUBLE DBL_MAX
#endif

#endif // DARWIN_COMPAT_H