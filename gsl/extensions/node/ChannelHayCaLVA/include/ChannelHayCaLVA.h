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

#ifndef ChannelHayCaLVA_H
#define ChannelHayCaLVA_H

#include "Lens.h"
#include "CG_ChannelHayCaLVA.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

class ChannelHayCaLVA : public CG_ChannelHayCaLVA
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayCaLVA();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
