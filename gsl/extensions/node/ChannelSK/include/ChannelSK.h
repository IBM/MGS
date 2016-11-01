#ifndef ChannelSK_H
#define ChannelSK_H

#include "Lens.h"
#include "CG_ChannelSK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "NTSMacros.h"

#if CHANNEL_SK == SK_TRAUB_1994
#define BASED_TEMPERATURE 25.0  // Celcius
#define Q10 2.3
#elif CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN || \
	  CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT
#define BASED_TEMPERATURE 25.0  // Celcius
#define Q10 2.3
#elif CHANNEL_SK == SK_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 2.3
#endif

#ifndef Q10
#define Q10 2.3  // default
#endif
class ChannelSK : public CG_ChannelSK
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelSK();

  private:
  dyn_var_t KV(dyn_var_t k, dyn_var_t d, dyn_var_t Vm);
  dyn_var_t fwrate(dyn_var_t v, dyn_var_t cai);
  dyn_var_t bwrate(dyn_var_t v, dyn_var_t cai);
};

#endif
