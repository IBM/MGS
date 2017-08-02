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

#ifndef ChannelHayHCN_H
#define ChannelHayHCN_H

#include "Lens.h"
#include "CG_ChannelHayHCN.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

class ChannelHayHCN : public CG_ChannelHayHCN
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayHCN();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
