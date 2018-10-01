#ifndef ChannelKv31_STR_FSI_mouse_H
#define ChannelKv31_STR_FSI_mouse_H

#include "Lens.h"
#include "CG_ChannelKv31_STR_FSI_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>

#define BASED_TEMPERATURE 25 // arbitrary
#define Q10 1 // sets Tadj =1

class ChannelKv31_STR_FSI_mouse : public CG_ChannelKv31_STR_FSI_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKv31_STR_FSI_mouse();
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif     
};

#endif
