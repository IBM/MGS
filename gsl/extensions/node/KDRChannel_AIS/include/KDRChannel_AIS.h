// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef KDRChannel_H
#define KDRChannel_H

#include "Mgs.h"
#include "CG_KDRChannel_AIS.h"
#include "rndm.h"

class KDRChannel_AIS : public CG_KDRChannel_AIS
{
   public:
      void update(RNG& rng);
      void initializeKDRChannels(RNG& rng);
      virtual ~KDRChannel_AIS();
   private:
};

#endif
