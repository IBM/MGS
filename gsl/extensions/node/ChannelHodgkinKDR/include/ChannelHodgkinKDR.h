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

#ifndef ChannelHodgkinKDR_H
#define ChannelHodgkinKDR_H

#include "Lens.h"
#include "CG_ChannelHodgkinKDR.h"
#include "rndm.h"

class ChannelHodgkinKDR : public CG_ChannelHodgkinKDR
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHodgkinKDR();
   private:
};

#endif
