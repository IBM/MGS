// Compatibility header for values.h on macOS
#ifndef VALUES_H
#define VALUES_H

#ifdef __APPLE__
// Include macOS equivalents
#include <limits.h>
#include <float.h>

// Define Linux values.h macros using macOS equivalents
#define MININT INT_MIN
#define MAXINT INT_MAX
#define MINFLOAT FLT_MIN
#ifndef MAXFLOAT
#define MAXFLOAT FLT_MAX
#endif
#define MINDOUBLE DBL_MIN
#define MAXDOUBLE DBL_MAX
#endif

#endif // VALUES_H
