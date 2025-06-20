// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

