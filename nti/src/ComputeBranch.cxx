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

#include "ComputeBranch.h"


Capsule& ComputeBranch::lastCapsule()
{
  assert(_nCapsules > 0);
  return _capsules[_nCapsules - 1];
}

dyn_var_t ComputeBranch::getSurfaceArea() {
  dyn_var_t rval=0;
  for (int ii=0; ii<_nCapsules; ++ii)
    rval+=_capsules[ii].getSurfaceArea();
  return rval;
}

