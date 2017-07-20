// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ChannelHayNat_H
#define ChannelHayNat_H

#include "Lens.h"
#include "CG_ChannelHayNat.h"
#include "rndm.h"

#include "../../nti/include/MaxComputeOrder.h"

class ChannelHayNat : public CG_ChannelHayNat
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayNat();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
