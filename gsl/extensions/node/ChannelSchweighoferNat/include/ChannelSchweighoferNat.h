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

#ifndef ChannelSchweighoferNat_H
#define ChannelSchweighoferNat_H

#include "Lens.h"
#include "CG_ChannelSchweighoferNat.h"
#include "rndm.h"

class ChannelSchweighoferNat : public CG_ChannelSchweighoferNat
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelSchweighoferNat();
   private:
      float vtrap(float x, float y);
};

#endif
