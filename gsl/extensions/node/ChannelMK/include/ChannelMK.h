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

#ifndef ChannelMK_H
#define ChannelMK_H

#include "Lens.h"
#include "CG_ChannelMK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
#define BASED_TEMPERATURE 21.0  // Celcius
#define Q10 3.0
#endif


class ChannelMK : public CG_ChannelMK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelMK();
   private:
};

#endif
