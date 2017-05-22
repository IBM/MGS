#ifndef ChannelNat_H
#define ChannelNat_H

#include "Lens.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_NAT == NAT_HODGKIN_HUXLEY_1952
#define BASED_TEMPERATURE 6.3  // Celcius
#define Q10 3.0
#elif CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.92
#elif CHANNEL_NAT == NAT_RUSH_RINZEL_1994
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_TRAUB_1994
#define BASED_TEMPERATURE 23  // Celcius
//TUAN TODO: maybe we need to update  all TRAUB model to 2.3
#define Q10 2.3
#elif CHANNEL_NAT == NAT_SCHWEIGHOFER_1999
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_COLBERT_PAN_2002
#define BASED_TEMPERATURE 23  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_WOLF_2005
#define BASED_TEMPERATURE 21.8  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_HAY_2011
//Modified Colbert - Pan (2002) model
#define BASED_TEMPERATURE 21  // Celcius
#define Q10 2.3
#endif

#ifndef Q10 
#define Q10 2.3 //default
#endif

class ChannelNat : public CG_ChannelNat
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelNat();
  static void initialize_others();

  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
#if CHANNEL_NAT == NAT_WOLF_2005 || \
    CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990
  const static dyn_var_t _Vmrange_taum[];
  const static dyn_var_t _Vmrange_tauh[];
  static dyn_var_t taumNat[];
  static dyn_var_t tauhNat[];
  static std::vector<dyn_var_t> Vmrange_taum;
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
};

#endif
