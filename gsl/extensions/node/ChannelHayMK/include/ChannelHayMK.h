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

#ifndef ChannelHayMK_H
#define ChannelHayMK_H

#include "Lens.h"
#include "CG_ChannelHayMK.h"
#include "rndm.h"

class ChannelHayMK : public CG_ChannelHayMK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayMK();
   private:
};

#endif
