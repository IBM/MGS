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

#ifndef KDRChannel_H
#define KDRChannel_H

#include "Lens.h"
#include "CG_KDRChannel.h"
#include "rndm.h"

class KDRChannel : public CG_KDRChannel
{
   public:
      void update(RNG& rng);
      void initializeKDRChannels(RNG& rng);
      virtual ~KDRChannel();
   private:
      float vtrap(float x, float y);
};

#endif
