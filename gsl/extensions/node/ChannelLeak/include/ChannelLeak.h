// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
*/


#ifndef ChannelLeak_H
#define ChannelLeak_H

#include "Lens.h"
#include "CG_ChannelLeak.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 


class ChannelLeak : public CG_ChannelLeak
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelLeak();
};

#endif
