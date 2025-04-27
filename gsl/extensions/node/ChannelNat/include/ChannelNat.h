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

#ifndef ChannelNat_H
#define ChannelNat_H

#include "Lens.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>

#if CHANNEL_NAT == NAT_HODGKIN_HUXLEY_1952
#define BASED_TEMPERATURE 6.3  // Celcius
#define Q10 3.0
#elif CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.92
#elif CHANNEL_NAT == NAT_RUSH_RINZEL_1994
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_TRAUB_1994
#define BASED_TEMPERATURE 23  // Celcius
//TUAN TODO: maybe we need to update  all TRAUB model to 2.3
#define Q10 2.3

#elif CHANNEL_NAT == NAT_WANG_BUSZAKI_1996
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.92 //To get a phi value = 5

#elif CHANNEL_NAT == NAT_SCHWEIGHOFER_1999
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 2.3

#elif CHANNEL_NAT == NAT_MAHON_2000
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.92 //To get a phi value = 5 at 37oC

#elif CHANNEL_NAT == NAT_COLBERT_PAN_2002
#define BASED_TEMPERATURE 23  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_WOLF_2005
#define BASED_TEMPERATURE 21.8  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_HAY_2011
//Modified Colbert - Pan (2002) model
#define BASED_TEMPERATURE 21  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_MSN_TUAN_JAMES_2017
#define BASED_TEMPERATURE 21.8  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_FUJITA_2012
#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1
#endif

#ifndef Q10
#define Q10 2.3 //default
#endif

//#define WRITE_GATES

class ChannelNat : public CG_ChannelNat
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelNat();
  static void initialize_others();

  private:
#if CHANNEL_NAT == NAT_WOLF_2005 || \
    CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990 || \
    CHANNEL_NAT == NAT_MSN_TUAN_JAMES_2017
  const static dyn_var_t _Vmrange_taum[];
  const static dyn_var_t _Vmrange_tauh[];
  static dyn_var_t taumNat[];
  static dyn_var_t tauhNat[];
  static std::vector<dyn_var_t> Vmrange_taum;
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
#if defined(WRITE_GATES)
  std::ofstream* outFile;
  float _prevTime;
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.05 // ms
#endif
};

#endif
