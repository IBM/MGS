#include "Lens.h"
#include "ChannelKIR.h"
#include "CG_ChannelKIR.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_KIR = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "KIR potassium current
//
#if CHANNEL_KIR == KIR_WOLF_2005
//  Kir2.1 
//  non-inactivating KIR current ~ KIR2 family (here is Kir2.1)
//     as found in substance P-expressing MSN in NAc [Mermelstein et al., 1998]
//  steady-state activation curve: use KIR channel (Kir2.1) data from 
//     human embryonic kidney cells [Kubo, Murata, 2001]
/* Mermelstein PG, Song WJ, Tkatch T, Yan Z, Surmeier DJ (1998) Inwardly
rectifying potassium (IRK) currents are correlated with IRK subunit
expression in rat nucleus accumbens medium spiny neurons. J Neurosci
18:6650-6661.

Uchimura N, Cherubini E, North RA (1989).  Inward rectification
in rat nucleus accumbens neurons. J Neurophysiol 62, 1280-1286.

Kubo Y, Murata Y (2001).  Control of rectification and permeation by two
distinct sites after the second transmembrane region in Kir2.1 K+
channel. J Physiol 531, 645-660.

Hayashi H, Fishman HM (1988). Inward rectifier K+ channel kinetics from
analysis of the complex conductance of aplysia neuronal membrane.
Biophys J 53, 747-757. 
*/
//  
//#define VHALF_M -52   
#define VHALF_M -82   // -52 - 30 = -82
#define k_M 13
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
    //dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKIR[index] * 2);
    dyn_var_t taum;
    if (index == 0)
      taum = taumKIR[0];
    else
      taum = linear_interp(Vmrange_taum[index-1], taumKIR[index-1], 
        Vmrange_taum[index], taumKIR[index], v);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
#else
    NOT IMPLEMENTED YET
#endif
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    g[i] = gbar[i] * m[i];
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelKIR::initialize(RNG& rng)
{
  pthread_once(&once_KIR, ChannelKIR::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels KIR Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      unsigned int j;
      //NOTE: 'n' bins are splitted by (n-1) points
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
          << "; gbar_dists.size = " << gbar_dists.size() << std::endl;
      }
      assert(gbar_values.size() -1 == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j];
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
#if CHANNEL_KIR == KIR_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
#else
    NOT IMPLEMENTED YET
// m[i] = am / (am + bm); //steady-state value
#endif
    g[i] = gbar[i] * m[i];
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelKIR::initialize_others()
{
#if CHANNEL_KIR == KIR_WOLF_2005
  std::vector<dyn_var_t> tmp(_Vmrange_taum, _Vmrange_taum + LOOKUP_TAUM_LENGTH);
  assert(sizeof(taumKIR) / sizeof(taumKIR[0]) == tmp.size());
	//Vmrange_taum.resize(tmp.size()-2);
  //for (unsigned long i = 1; i < tmp.size() - 1; i++)
  //  Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  Vmrange_taum = tmp;
#endif
}

ChannelKIR::~ChannelKIR() {}
