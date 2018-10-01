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


#ifndef ChannelHCN_H
#define ChannelHCN_H

#include "Lens.h"
#include "CG_ChannelHCN.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>


#if CHANNEL_HCN == HCN_KOLE_2006
#define BASED_TEMPERATURE 23  // Celcius
#define Q10 3.0
#elif CHANNEL_HCN == HCN_HUGUENARD_MCCORMICK_1992
#define BASED_TEMPERATURE 35.5  // Celcius
#define Q10 3.0
#elif CHANNEL_HCN == HCN_KOLE_2006 || \
	  CHANNEL_HCN == HCN_HAY_2011
#define BASED_TEMPERATURE 35  // Celcius
#define Q10 3.0

#elif CHANNEL_HCN == HCN_FUJITA_2012 
#define BASED_TEMPERATURE 25 //arbitrary
#define Q10 		1 // set Tadj = 1
#endif

//default
#ifndef BASED_TEMPERATURE
#define BASED_TEMPERATURE 35  // Celcius
#endif
#ifndef Q10
#define Q10 3.0  // default
#endif

class ChannelHCN : public CG_ChannelHCN
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelHCN();
  dyn_var_t conductance(int i);

  private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        
};

#endif
