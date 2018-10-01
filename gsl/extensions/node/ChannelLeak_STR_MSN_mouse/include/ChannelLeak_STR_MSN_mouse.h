// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelLeak_STR_MSN_mouse_H
#define ChannelLeak_STR_MSN_mouse_H

#include "Lens.h"
#include "CG_ChannelLeak_STR_MSN_mouse.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 

class ChannelLeak_STR_MSN_mouse : public CG_ChannelLeak_STR_MSN_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelLeak_STR_MSN_mouse();
  private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        


};

#endif
