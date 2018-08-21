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

#ifndef KDRChannel_IO_H
#define KDRChannel_IO_H

#include "Lens.h"
#include "CG_KDRChannel_IO.h"
#include "rndm.h"

class KDRChannel_IO : public CG_KDRChannel_IO
{
   public:
      void update(RNG& rng);
      void initializeKDRChannels(RNG& rng);
      virtual ~KDRChannel_IO();
   private:
};

#endif
