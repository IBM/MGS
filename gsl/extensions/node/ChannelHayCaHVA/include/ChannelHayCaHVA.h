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
};

#endif
