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

#ifndef NaChannel_AIS_H
#define NaChannel_AIS_H

#include "Lens.h"
#include "CG_NaChannel_AIS.h"
#include "rndm.h"

class NaChannel_AIS : public CG_NaChannel_AIS
{
   public:
      void update(RNG& rng);
      void initializeNaChannels(RNG& rng);
      virtual ~NaChannel_AIS();
   private:
      float vtrap(float x, float y);
};

#endif
