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

#ifndef ChannelSchweighoferKDR_H
#define ChannelSchweighoferKDR_H

#include "Lens.h"
#include "CG_ChannelSchweighoferKDR.h"
#include "rndm.h"

class ChannelSchweighoferKDR : public CG_ChannelSchweighoferKDR
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelSchweighoferKDR();
   private:
      float vtrap(float x, float y);
};

#endif
