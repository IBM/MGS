#ifndef ChannelNat_AIS_H
#define ChannelNat_AIS_H

#include "Lens.h"
#include "CG_ChannelNat_AIS.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
#define BASED_TEMPERATURE 25  // Celcius
#define Q10 3.0
#elif CHANNEL_NAT_AIS == NAT_AIS_MSN_TUAN_JAMES_2017
#define BASED_TEMPERATURE 21.8  // Celcius
#define Q10 2.3
#endif

class ChannelNat_AIS : public CG_ChannelNat_AIS
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelNat_AIS();
      static void initialize_others();
   private:
#if CHANNEL_NAT_AIS == NAT_AIS_MSN_TUAN_JAMES_2017
  const static dyn_var_t _Vmrange_taum[];
  const static dyn_var_t _Vmrange_tauh[];
  static dyn_var_t taumNat[];
  static dyn_var_t tauhNat[];
  static std::vector<dyn_var_t> Vmrange_taum;
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
};

#endif
