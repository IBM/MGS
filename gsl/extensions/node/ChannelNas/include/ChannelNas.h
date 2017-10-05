#ifndef ChannelNas_H
#define ChannelNas_H

#include "Lens.h"
#include "CG_ChannelNas.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_NAS == NAS_MAHON_2000        
#define BASED_TEMPERATURE 22.0  // Celcius 
#define Q10 2.5                            

#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ChannelNas : public CG_ChannelNas
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelNas();
      static void initialize_others();//new

  private:
//#if CHANNEL_NAS == NAS_WOLF_2005
//	const static dyn_var_t _Vmrange_tauh[];
//	static dyn_var_t tauhNap[];
//	static std::vector<dyn_var_t> Vmrange_tauh;
//#endif
};

#endif
