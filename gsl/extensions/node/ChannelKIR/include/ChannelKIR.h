#ifndef ChannelKIR_H
#define ChannelKIR_H

#include "Lens.h"
#include "CG_ChannelKIR.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KIR  == KIR_HAYASHI_FISHMAN_1988
//experimental data
#define BASED_TEMPERATURE 23.0  // Celcius -22-24
#define Q10 1.8                                        

#elif CHANNEL_KIR  == KIR_MAHON_2000                   
#define BASED_TEMPERATURE 22.0  // Celcius - in vitro  
#define Q10 2.5                                        

#elif CHANNEL_KIR  == KIR_WOLF_2005
#if 0 // USE_NEURON_CODE == 1  
#define BASED_TEMPERATURE 43.320  // Celcius - in vitro
#define Q10 2.3    //the temperature above was set to return qfact=0.5
#else
#define BASED_TEMPERATURE 35.0  // Celcius - in vitro
#define Q10 2.3    
#endif

#elif CHANNEL_KIR  == KIR_STEEPHEN_MANCHANDA_2009
#define BASED_TEMPERATURE 35.0  // Celcius - in vitro
#define Q10 2.3    

#elif CHANNEL_KIR  == KIR2_1_STEEPHEN_MANCHANDA_2009
#define BASED_TEMPERATURE 35.0  // Celcius - in vitro
#define Q10 2.3    

#elif CHANNEL_KIR  == KIR2_1_TUAN_JAMES_2017
#define BASED_TEMPERATURE 35.0  // Celcius - in vitro
#define Q10 2.3    

#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ChannelKIR : public CG_ChannelKIR
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKIR();
  static void initialize_others();  // new
  private:
#if CHANNEL_KIR == KIR_HAYASHI_FISHMAN_1988 || \
  CHANNEL_KIR == KIR_WOLF_2005 || \
  CHANNEL_KIR == KIR_STEEPHEN_MANCHANDA_2009 || \
  CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
  CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
  const static dyn_var_t _Vmrange_taum[];
  static dyn_var_t taumKIR[];
  static std::vector<dyn_var_t> Vmrange_taum;
#if CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
    CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
  static dyn_var_t h_inf_KIR[];
  const static dyn_var_t _Vmrange_tauh[];
  static dyn_var_t tauhKIR[];
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
#endif
};

#endif
