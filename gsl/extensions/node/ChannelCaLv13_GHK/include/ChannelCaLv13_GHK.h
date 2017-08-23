#ifndef ChannelCaLv13_GHK_H
#define ChannelCaLv13_GHK_H

#include "CG_ChannelCaLv13_GHK.h"
#include "Lens.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_CaLv13 == CaLv13_GHK_WOLF_2005
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif

class ChannelCaLv13_GHK : public CG_ChannelCaLv13_GHK
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  static void initialize_others();
  virtual ~ChannelCaLv13_GHK();

  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
  dyn_var_t update_current(dyn_var_t v, dyn_var_t cai, int i);
};

#endif
