// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef tangent_h
#define tangent_h

#include <math.h>
#include <cassert>

inline bool intraSegmentSphere(double* cds1, double* cds2, double* cdsPrime)
{
  return false;
}

inline void extraSegmentSphere(double* cds1, double* cds2, double* cds3, double* cdsPrime)
{
}

inline double angle(double* cds1, double* cds2, double* cds3)
{
  return 0.0;
}

#endif
