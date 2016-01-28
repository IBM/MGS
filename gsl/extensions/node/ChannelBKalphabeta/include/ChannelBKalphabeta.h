#ifndef ChannelBKalphabeta_H
#define ChannelBKalphabeta_H

#include "Lens.h"
#include "CG_ChannelBKalphabeta.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_BKalphabeta == BKalphabeta_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelBKalphabeta : public CG_ChannelBKalphabeta
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelBKalphabeta();

  private:
  dyn_var_t alpha(dyn_var_t tmin, dyn_var_t tmax, dyn_var_t v, dyn_var_t vhalf,
                  dyn_var_t k);
  dyn_var_t alp(dyn_var_t tmin, dyn_var_t v, dyn_var_t vhalf, dyn_var_t k);
};

#endif
