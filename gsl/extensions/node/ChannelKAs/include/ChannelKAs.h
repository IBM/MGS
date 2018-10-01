/*
=================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-03-25-2018

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

================================================================

(C) Copyright 2018 New Jersey Institute of Technology.

=================================================================
*/


#ifndef ChannelKAs_H
#define ChannelKAs_H

#include "Lens.h"
#include "CG_ChannelKAs.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>

#if CHANNEL_KAs == KAs_WOLF_2005
#define BASED_TEMPERATURE 15.0  // Celcius
#define Q10 3.0

#elif CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
#define BASED_TEMPERATURE 21.0  // Celcius
#define Q10 2.3

#elif CHANNEL_KAs == KAs_MAHON_2000        
#define BASED_TEMPERATURE 22.0  // Celcius 
#define Q10 2.5                            

#elif CHANNEL_KAs == KAs_EVANS_2012
#define BASED_TEMPERATURE 35.0  // Celcius 
#define Q10 2.5                            

#elif CHANNEL_KAs == KAs_FUJITA_2012
#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1


#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelKAs : public CG_ChannelKAs
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKAs();
  static void initialize_others();  // new
  private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        
};

#endif
