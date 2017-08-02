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

#ifndef KDRChannel_H
#define KDRChannel_H

#include "Lens.h"
#include "CG_KDRChannel_AIS.h"
#include "rndm.h"

class KDRChannel_AIS : public CG_KDRChannel_AIS
{
   public:
      void update(RNG& rng);
      void initializeKDRChannels(RNG& rng);
      virtual ~KDRChannel_AIS();
   private:
      float vtrap(float x, float y);
};

#endif
