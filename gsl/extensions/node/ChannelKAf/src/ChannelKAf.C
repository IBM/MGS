#include "Lens.h"
#include "ChannelKAf.h"
#include "CG_ChannelKAf.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

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
    g[i] = gbar[i] * m[i] * m[i] * h[i];
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
  for (unsigned i = 0; i < size; ++i)
  {
    gbar[i] = gbar[0];
    dyn_var_t v = (*V)[i];
#if CHANNEL_KAf == KAf_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
#else
    NOT IMPLEMENTED YET;
// m[i] = am / (am + bm);  // steady-state value
// h[i] = ah / (ah + bh);
#endif
    g[i] = gbar[i] * m[i] * m[i] * h[i];
  }
}

void ChannelKAf::initialize_others()
{
#if CHANNEL_KAf == KAf_WOLF_2005
  std::vector<dyn_var_t> tmp(_Vmrange_taum, _Vmrange_taum + LOOKUP_TAUM_LENGTH);
  assert(sizeof(taumKAf) / sizeof(taumKAf[0]) == tmp.size());
  for (int i = 1; i < tmp.size() - 1; i++)
    Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
#endif
}
ChannelKAf::~ChannelKAf() {}
