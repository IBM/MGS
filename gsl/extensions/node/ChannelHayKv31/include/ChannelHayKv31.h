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

#ifndef ChannelHayKv31_H
#define ChannelHayKv31_H

#include "Lens.h"
#include "CG_ChannelHayKv31.h"
#include "rndm.h"

class ChannelHayKv31 : public CG_ChannelHayKv31
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayKv31();
   private:
};

#endif
