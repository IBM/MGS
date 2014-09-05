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

#ifndef KCaChannel_H
#define KCaChannel_H

#include "Lens.h"
#include "CG_KCaChannel.h"
#include "rndm.h"

class KCaChannel : public CG_KCaChannel
{
   public:
      void update(RNG& rng);
      void initializeKCaChannels(RNG& rng);
      virtual ~KCaChannel();
};

#endif
