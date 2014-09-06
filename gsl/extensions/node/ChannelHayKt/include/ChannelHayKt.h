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

#ifndef ChannelHayKt_H
#define ChannelHayKt_H

#include "Lens.h"
#include "CG_ChannelHayKt.h"
#include "rndm.h"

class ChannelHayKt : public CG_ChannelHayKt
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayKt();
   private:
      float vtrap(float x, float y);
};

#endif
