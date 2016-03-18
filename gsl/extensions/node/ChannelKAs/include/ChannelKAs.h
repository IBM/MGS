#ifndef ChannelKAs_H
#define ChannelKAs_H

#include "Lens.h"
#include "CG_ChannelKAs.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KAs == KAs_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelKAs : public CG_ChannelKAs
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKAs();
  static void initialize_others();  // new
  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);  // new
};

#endif
