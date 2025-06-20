// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
// 
// =============================================================================


#ifndef ChannelCaHVA_H
#define ChannelCaHVA_H

#include "Mgs.h"
#include "CG_ChannelCaHVA.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 


#if CHANNEL_CaHVA == CaHVA_REUVENI_AMITAI_GUTNICK_1993 
//note: DUAL_GATE means 2 gates are used: activate + inactivate
#define DUAL_GATE _YES 
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif CHANNEL_CaHVA == CaHVA_TRAUB_1994  //Temperature is not being used
#define DUAL_GATE _NO
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif CHANNEL_CaHVA == CaHVA_FUJITA_2012
#define BASED_TEMPERATURE 30 //arbitrary
#define Q10 1 // sets Tadj =1
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelCaHVA : public CG_ChannelCaHVA
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~ChannelCaHVA();
   private:

#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        

      
};

#endif
