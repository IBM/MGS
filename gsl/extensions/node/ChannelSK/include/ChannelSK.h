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

#ifndef ChannelSK_H
#define ChannelSK_H

#include "Mgs.h"
#include "CG_ChannelSK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "NTSMacros.h"
#include "SegmentDescriptor.h"
#include <fstream>


#if CHANNEL_SK == SK_TRAUB_1994
#define BASED_TEMPERATURE 25.0  // Celcius
#define Q10 2.3
#elif CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN || \
	  CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT
#define BASED_TEMPERATURE 25.0  // Celcius
#define Q10 2.3
#elif CHANNEL_SK == SK_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 2.3
#elif CHANNEL_SK == SK_FUJITA_2012
#define BASED_TEMPERATURE 25.0 // arbitrary
#define Q10 1 // sets Tadj to 1
#endif

#ifndef Q10
#define Q10 2.3  // default
#endif
class ChannelSK : public CG_ChannelSK
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelSK();

  private:
  dyn_var_t KV(dyn_var_t k, dyn_var_t d, dyn_var_t Vm);
  dyn_var_t fwrate(dyn_var_t v, dyn_var_t cai);
  dyn_var_t bwrate(dyn_var_t v, dyn_var_t cai);
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif 
};

#endif
