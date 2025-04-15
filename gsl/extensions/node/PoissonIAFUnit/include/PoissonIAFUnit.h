// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PoissonIAFUnit_H
#define PoissonIAFUnit_H

#include "Lens.h"
#include "CG_PoissonIAFUnit.h"
#include "rndm.h"

class PoissonIAFUnit : public CG_PoissonIAFUnit
{
 public:
  void update(RNG& rng);
  virtual ~PoissonIAFUnit();
};

#endif
