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

#ifndef ChannelHayNap_H
#define ChannelHayNap_H

#include "../../nti/include/MaxComputeOrder.h"

#include "Lens.h"
#include "CG_ChannelHayNap.h"
#include "rndm.h"

class ChannelHayNap : public CG_ChannelHayNap
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayNap();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
