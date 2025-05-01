// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

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