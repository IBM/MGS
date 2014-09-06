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

#ifndef ChannelHayCaHVA_H
#define ChannelHayCaHVA_H

#include "Lens.h"
#include "CG_ChannelHayCaHVA.h"
#include "rndm.h"

class ChannelHayCaHVA : public CG_ChannelHayCaHVA
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayCaHVA();
   private:
      float vtrap(float x, float y);
};

#endif
