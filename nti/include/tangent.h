// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
