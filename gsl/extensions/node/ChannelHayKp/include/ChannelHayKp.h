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

#ifndef ChannelHayKp_H
#define ChannelHayKp_H

#include "Lens.h"
#include "CG_ChannelHayKp.h"
#include "rndm.h"

class ChannelHayKp : public CG_ChannelHayKp
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayKp();
   private:
};

#endif
