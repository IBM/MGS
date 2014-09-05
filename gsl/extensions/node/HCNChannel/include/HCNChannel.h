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

#ifndef HCNChannel_H
#define HCNChannel_H

#include "Lens.h"
#include "CG_HCNChannel.h"
#include "rndm.h"

class HCNChannel : public CG_HCNChannel
{
   public:
      void update(RNG& rng);
      void initializeHCNChannels(RNG& rng);
      virtual ~HCNChannel();
   private:
      float vtrap(float x, float y);
};

#endif
