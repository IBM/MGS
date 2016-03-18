#ifndef ChannelCaPQ_GHK_H
#define ChannelCaPQ_GHK_H

#include "Lens.h"
#include "CG_ChannelCaPQ_GHK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_CaPQ == CaPQ_GHK_WOLF_2005
#define BASED_TEMPERATURE 35.0 //Celcius
#define Q10 3.0
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif

class ChannelCaPQ_GHK : public CG_ChannelCaPQ_GHK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
			static void initialize_others();
      virtual ~ChannelCaPQ_GHK();
	 private:
			dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
