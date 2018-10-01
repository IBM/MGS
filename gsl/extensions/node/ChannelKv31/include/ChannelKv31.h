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


#ifndef ChannelKv31_H
#define ChannelKv31_H

#include "Lens.h"
#include "CG_ChannelKv31.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 


#if CHANNEL_Kv31 == Kv31_RETTIG_1992 
#define BASED_TEMPERATURE 23.0  // Celcius
#define Q10 3.0
#elif CHANNEL_Kv31 == Kv31_FUJITA_2012
#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1
#endif

class ChannelKv31 : public CG_ChannelKv31
{
  public:
    void update(RNG& rng);
    void initialize(RNG& rng);
    virtual ~ChannelKv31();
  private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        


};

#endif
