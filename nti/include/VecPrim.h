// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
