#ifndef ChannelCaHVA_H
#define ChannelCaHVA_H

#include "Lens.h"
#include "CG_ChannelCaHVA.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_CaHVA == CaHVA_REUVENI_AMITAI_GUTNICK_1993 
//note: DUAL_GATE means 2 gates are used: activate + inactivate
#define DUAL_GATE _YES 
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif CHANNEL_CaHVA == CaHVA_TRAUB_1994  //Temperature is not being used
#define DUAL_GATE _NO
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelCaHVA : public CG_ChannelCaHVA
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~ChannelCaHVA();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
