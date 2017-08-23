#ifndef ChannelCaLv12_GHK_H
#define ChannelCaLv12_GHK_H

#include "CG_ChannelCaLv12_GHK.h"
#include "Lens.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_CaLv12 == CaLv12_GHK_WOLF_2005
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#elif CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 2.3
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelCaLv12_GHK : public CG_ChannelCaLv12_GHK
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  static void initialize_others();
  virtual ~ChannelCaLv12_GHK();

  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
  dyn_var_t update_current(dyn_var_t v, dyn_var_t cai, int i);
};

#endif
