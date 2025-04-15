// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
