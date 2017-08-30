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

#ifndef CalChannel_H
#define CalChannel_H

#include "Lens.h"
#include "CG_CalChannel.h"
#include "rndm.h"

class CalChannel : public CG_CalChannel
{
   public:
      void update(RNG& rng);
      void initializeCalChannels(RNG& rng);
      virtual ~CalChannel();
   private:
};

#endif
