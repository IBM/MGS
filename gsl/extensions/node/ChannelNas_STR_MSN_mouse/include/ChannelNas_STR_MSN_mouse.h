// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelNas_STR_MSN_mouse_H
#define ChannelNas_STR_MSN_mouse_H

#include "Lens.h"
#include "CG_ChannelNas_STR_MSN_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#define BASED_TEMPERATURE 22.0  // Celcius 
#define Q10 2.5                            

class ChannelNas_STR_MSN_mouse : public CG_ChannelNas_STR_MSN_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelNas_STR_MSN_mouse();
};

#endif
