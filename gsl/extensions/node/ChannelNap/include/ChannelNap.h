/* =================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-07-18-2017

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

=================================================================

(C) Copyright 2018 New Jersey Institute of Technology.

=================================================================
*/


#ifndef ChannelNap_H
#define ChannelNap_H

#include "Lens.h"
#include "CG_ChannelNap.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>


#if CHANNEL_NAP == NAP_WOLF_2005
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#elif CHANNEL_NAP == NAP_MAGISTRETTI_1999
//NOTE: Hay et al. (2011) also use this model
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#elif CHANNEL_NAP == NAP_MAHON_2000        
#define BASED_TEMPERATURE 22.0  // Celcius 
#define Q10 2.5                            

#elif CHANNEL_NAP == NAP_FUJITA_2012
#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // set Tadj =1 
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ChannelNap : public CG_ChannelNap
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelNap();

  static void initialize_others();//new

  private:
#if CHANNEL_NAP == NAP_WOLF_2005
	const static dyn_var_t _Vmrange_tauh[];
	static dyn_var_t tauhNap[];
	static std::vector<dyn_var_t> Vmrange_tauh;
#endif
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        
};

#endif
