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


#ifndef ChannelKAf_H
#define ChannelKAf_H

#include "CG_ChannelKAf.h"
#include "Lens.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 


#if CHANNEL_KAf == KAf_TRAUB_1994  // There is no temperature dependence
#define BASED_TEMPERATURE 23.0     // Celcius
#define Q10 3.0

#elif CHANNEL_KAf == KAf_KORNGREEN_SAKMANN_2000
#define BASED_TEMPERATURE 21.0  // Celcius
#define Q10 2.3

#elif CHANNEL_KAf == KAf_MAHON_2000           
#define BASED_TEMPERATURE 22.0 // Celcius     
#define Q10 2.5                               

#elif CHANNEL_KAf == KAf_WOLF_2005
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3

#elif CHANNEL_KAf == KAf_EVANS_2012
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3

#elif CHANNEL_KAf == KAf_FUJITA_2012
#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1

#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelKAf : public CG_ChannelKAf
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKAf();
  static void initialize_others();  // new
  private:
#if CHANNEL_KAf == KAf_WOLF_2005
  const static dyn_var_t _Vmrange_taum[];
  static dyn_var_t taumKAf[];
  static std::vector<dyn_var_t> Vmrange_taum;
#endif
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        


};

#endif
