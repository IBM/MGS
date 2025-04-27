// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
// 
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
// 
// =============================================================================

#ifndef ChannelMK_H
#define ChannelMK_H

#include "Lens.h"
#include "CG_ChannelMK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>


#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
#define BASED_TEMPERATURE 21.0  // Celcius
#define Q10 3.0
#elif CHANNEL_MK == MK_FUJITA_2012
#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1

#endif


class ChannelMK : public CG_ChannelMK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelMK();
   private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif   
};

#endif
