// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelSK_GPe_mouse_H
#define ChannelSK_GPe_mouse_H

#include "Lens.h"
#include "CG_ChannelSK_GPe_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "NTSMacros.h"
#include "SegmentDescriptor.h"
#include <fstream>

#define BASED_TEMPERATURE 25.0 // arbitrary
#define Q10 1 // sets Tadj to 1

class ChannelSK_GPe_mouse : public CG_ChannelSK_GPe_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelSK_GPe_mouse();
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
