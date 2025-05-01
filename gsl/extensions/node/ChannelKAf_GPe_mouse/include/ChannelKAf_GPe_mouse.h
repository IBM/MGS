// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKAf_GPe_mouse_H
#define ChannelKAf_GPe_mouse_H

#include "Mgs.h"
#include "CG_ChannelKAf_GPe_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 

#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1


class ChannelKAf_GPe_mouse : public CG_ChannelKAf_GPe_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKAf_GPe_mouse();
  private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        


};

#endif
