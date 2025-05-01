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

#ifndef ChannelKDR_H
#define ChannelKDR_H

#include "Mgs.h"
#include "CG_ChannelKDR.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 

#if CHANNEL_KDR == KDR_HODGKIN_HUXLEY_1952
#define BASED_TEMPERATURE 6.3  // Celcius
#define Q10 3.0
#elif CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif CHANNEL_KDR == KDR_TRAUB_1994 || \
      CHANNEL_KDR == KDR_TRAUB_1995 //not being used for these models
#define BASED_TEMPERATURE 23.0
#define Q10 3.0

#elif CHANNEL_KDR == KDR_MAHON_2000 || \
      CHANNEL_KDR == KDR_WANG_BUSZAKI_1996                                    
#define BASED_TEMPERATURE 22.0  // Celcius                                 
#define Q10 2.92 //To get a phi value equivalent to 5 as used in the model 

#elif CHANNEL_KDR == KDR_MIGLIORE_1999
#define BASED_TEMPERATURE 35.0  // Celcius                                 
#define Q10 2.92 

#elif CHANNEL_KDR == KDR_ERISIR_1999 || \
      CHANNEL_KDR == KDR_TUAN_JAMES_2017
#define BASED_TEMPERATURE 23.0  // Celcius                                 
#define Q10 2.92

#elif CHANNEL_KDR == KDR_FUJITA_2012
#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1

#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif

class ChannelKDR : public CG_ChannelKDR
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKDR();
   private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        

};

#endif
