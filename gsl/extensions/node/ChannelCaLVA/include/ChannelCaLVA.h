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
// =================================================================

#ifndef ChannelCaLVA_H
#define ChannelCaLVA_H

#include "Lens.h"
#include "CG_ChannelCaLVA.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

#if CHANNEL_CaLVA == CaLVA_HAY_2011
#define DUAL_GATE _YES 
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#endif

class ChannelCaLVA : public CG_ChannelCaLVA
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelCaLVA();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
