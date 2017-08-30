// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef NaChannel_H
#define NaChannel_H

#include "Lens.h"
#include "CG_NaChannel.h"
#include "rndm.h"

#include "../../nti/include/MaxComputeOrder.h"

class NaChannel : public CG_NaChannel
{
   public:
      void update(RNG& rng);
      void initializeNaChannels(RNG& rng);
      virtual ~NaChannel();
   private:
};

#endif
