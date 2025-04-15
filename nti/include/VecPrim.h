// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _INCLUDED_VECPRIM__
#define _INCLUDED_VECPRIM__

#include <cmath>
#include <cassert>

static inline
double SqDist(double *A, double *B)
{
  double d = 0;
  for(int i = 0; i<3; ++i)
    {
      double dx = B[i] - A[i];
      d += dx * dx;
    }
  return d;
}

static inline
double Vec3Dot(double *A, double *B)
{
  double d = 0;
  for(int i = 0;i < 3; i++)
    {
      d += B[i] * A[i];
    }
  return d;
}

static inline
double Angle3(double* A, double* B, double* C)
{
  double a=SqDist(B,C);
  double c=SqDist(A,B);
  double rval= acos((SqDist(A,C)-c-a)/(-2.0*sqrt(c*a)));
  assert(!std::isnan(rval));
  return rval;
}
#endif
