#include "Lens.h"
#include "ChannelKIR.h"
#include "CG_ChannelKIR.h"
#include "rndm.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_KIR = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "KIR potassium current
//
#if CHANNEL_KIR == KIR_WOLF_2005
#define VHALF_M -13.5
#define k_M -11.8
#define LOOKUP_TAUM_LENGTH 16
const dyn_var_t ChannelKIR::_Vmrange_taum[] = {
    -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50};
dyn_var_t ChannelKIR::taumKIR[] = {
    3.7313, 4.0000, 4.7170, 5.3763, 6.0606, 6.8966, 7.6923, 7.1429,
    5.8824, 4.4444, 4.0000, 4.0000, 4.0000, 4.0000, 4.0000, 4.0000};
std::vector<dyn_var_t> ChannelKIR::Vmrange_taum;
#else
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelKIR::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelKIR::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KIR == KIR_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKIR[index] * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
#else
    NOT IMPLEMENTED YET
#endif
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    g[i] = gbar[i] * m[i];
  }
}

void ChannelKIR::initialize(RNG& rng)
{
  pthread_once(&once_KIR, ChannelKIR::initialize_others);
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
  // initialize
  for (unsigned i = 0; i < size; ++i)
  {
    gbar[i] = gbar[0];
    dyn_var_t v = (*V)[i];
#if CHANNEL_KIR == KIR_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
#else
    NOT IMPLEMENTED YET
// m[i] = am / (am + bm); //steady-state value
#endif
    g[i] = gbar[i] * m[i];
  }
}

void ChannelKIR::initialize_others()
{
#if CHANNEL_KIR == KIR_WOLF_2005
  std::vector<dyn_var_t> tmp(_Vmrange_taum, _Vmrange_taum + LOOKUP_TAUM_LENGTH);
  assert(sizeof(taumKIR) / sizeof(taumKIR[0]) == tmp.size());
  for (int i = 1; i < tmp.size() - 1; i++)
    Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
#endif
}

ChannelKIR::~ChannelKIR() {}
