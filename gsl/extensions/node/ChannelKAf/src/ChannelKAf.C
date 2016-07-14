#include "Lens.h"
#include "ChannelKAf.h"
#include "CG_ChannelKAf.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#include "SegmentDescriptor.h"

static pthread_once_t once_KAf = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "KAf potassium current
//
#if CHANNEL_KAf == KAf_WOLF_2005
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -10.0
#define k_M -17.7
#define VHALF_H -75.6
#define k_H 10.0
#define LOOKUP_TAUM_LENGTH 11  // size of the below array
const dyn_var_t ChannelKAf::_Vmrange_taum[] = {-40, -30, -20, -10, 0, 10,
                                               20,  30,  40,  50,  60};
dyn_var_t ChannelKAf::taumKAf[] = {1.8, 1.1, 1.0, 1.0, 0.9, 0.8,
                                   0.9, 0.9, 0.9, 0.8, 0.8};
std::vector<dyn_var_t> ChannelKAf::Vmrange_taum;
//The time constants in Traub's models is ~25 ms typically KAf ~ 30ms
#elif CHANNEL_KAf == KAf_TRAUB_1994
#define AMC -0.02
#define AMV 13.1
#define AMD -10.0
#define BMC 0.0175
#define BMV 40.1
#define BMD 10.0
#define AHC 0.0016
#define AHV -13.0
#define AHD -18
#define BHC 0.05
#define BHV 60.1
#define BHD -5.0


#else
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelKAf::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelKAf::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KAf == KAf_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKAf[index] * 2);
    const dyn_var_t tau_h = 4.67;
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);

#elif CHANNEL_KAf == KAf_TRAUB_1994
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    // Traub Models do not have temperature dependence and hence Tadj is not used
    dyn_var_t pm = 0.5 * dt * (am + bm) ;
    m[i] = (dt * am  + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) ;
    h[i] = (dt * ah  + h[i] * (1.0 - ph)) / (1.0 + ph);




#else
    NOT IMPLEMENTED YET
#endif
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep m in [0, 1]
    if (h[i] < 0.0)
    {
      h[i] = 0.0;
    }
    else if (h[i] > 1.0)
    {
      h[i] = 1.0;
    }
    
#if CHANNEL_KAf == KAf_TRAUB_1994
     g[i] = gbar[i] * m[i] * h[i];
#else
     g[i] = gbar[i] * m[i] * m[i] * h[i];
#endif
  }
}

void ChannelKAf::initialize(RNG& rng)
{
  pthread_once(&once_KAf, ChannelKAf::initialize_others);
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
#endif
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      unsigned int j;
      assert(gbar_values.size() == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    } 
		/*else if (gbar_values.size() == 1) {
      gbar[i] = gbar_values[0];
    } */
		else if (gbar_branchorders.size() > 0)
		{
      unsigned int j;
      assert(gbar_values.size() == gbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j=0; j<gbar_branchorders.size(); ++j) {
        if (segmentDescriptor.getBranchOrder(branchData->key) == gbar_branchorders[j]) break;
      }
			if (j == gbar_branchorders.size() and gbar_branchorders[j-1] == GlobalNTS::anybranch_at_end)
			{
				gbar[i] = gbar_values[j-1];
			}
			else if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
		}
		else {
      gbar[i] = gbar_default;
    }
	}
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KAf == KAf_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));

#elif CHANNEL_KAf == KAf_TRAUB_1994    
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
#else
    NOT IMPLEMENTED YET;
// m[i] = am / (am + bm);  // steady-state value
// h[i] = ah / (ah + bh);
#endif

#if CHANNEL_KAf == KAf_TRAUB_1994
    g[i] = gbar[i] * m[i] * h[i];
#else
    g[i] = gbar[i] * m[i] * m[i] * h[i];
#endif 
 }
}

void ChannelKAf::initialize_others()
{
#if CHANNEL_KAf == KAf_WOLF_2005
  std::vector<dyn_var_t> tmp(_Vmrange_taum, _Vmrange_taum + LOOKUP_TAUM_LENGTH);
  assert(sizeof(taumKAf) / sizeof(taumKAf[0]) == tmp.size());
	Vmrange_taum.resize(tmp.size()-2);
  for (unsigned long i = 1; i < tmp.size() - 1; i++)
    Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
#endif
}

ChannelKAf::~ChannelKAf() {}
