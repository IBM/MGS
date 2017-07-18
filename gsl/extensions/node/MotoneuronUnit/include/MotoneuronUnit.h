// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef MotoneuronUnit_H
#define MotoneuronUnit_H

#include "Lens.h"
#include "CG_MotoneuronUnit.h"
#include "rndm.h"

class MotoneuronUnit : public CG_MotoneuronUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~MotoneuronUnit();
};

#endif
