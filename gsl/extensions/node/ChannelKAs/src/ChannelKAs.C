#include "Lens.h"
#include "ChannelKAs.h"
#include "CG_ChannelKAs.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_KAs = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "KAs potassium current
//
#if CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
// Korngreen - Sakmann (2000) Vm-gated K+ channels in PL5 neocortical young rat
// recorded in soma using nucleated outside-out patches
//          ... dendrite up to 430 um from soma using cell-attached recording
// NOTE: estimated patch surface area 440+/- 10 um^2
//             average series resistance 13.2+/-0.7 M.Ohm
//                    input resistance 2.7+/- 0.2 G.Ohm
//                    capacitance      2.2+/- 0.1 pF
#define IMV 11.0
#define IMD 12.0
#define IHV 64.0
#define IHD 11.0
#define TMC 1.25
#define TMF1 175.03
#define TMF2 13.0
#define TMV 10.0
#define TMD1 0.026
#define TMD2 -0.026
#define THC1 360.0
#define THC2 1010.0
#define THF 24.0
#define THV1 65.0
#define THV2 85.0
#define THD 48.0
//#define T_ADJ 2.9529 // 2.3^((34-21)/10)

#elif CHANNEL_KAs == KAs_WOLF_2005
//  Inactivation from
//  Activation from
//    1. Shen et al. (2004)
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -27.0
#define k_M -16.0
#define VHALF_H -33.5
#define k_H 21.5
#define frac_inact 0.996  // 'a' term
#define AHC 1.0
#define AHV 90.96
#define AHD 29.01
#define BHC 1.0
#define BHV 90.96
#define BHD -100
#else
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelKAs::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelKAs::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
    { 
    dyn_var_t minf = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    dyn_var_t taum;
    if (v < -60.0) {
      //taum = (TMC + TMF1*exp(TMD1*(v + TMV)))/T_ADJ;
      taum = (TMC + TMF1*exp(TMD1*(v + TMV)))/getSharedMembers().Tadj;
    } else {
      //taum = (TMC + TMF2*exp(TMD2*(v + TMV)))/T_ADJ;
      taum = (TMC + TMF2*exp(TMD2*(v + TMV)))/getSharedMembers().Tadj;
    }
    dyn_var_t hinf = 1.0/(1.0 + exp((v + IHV)/IHD));
    //dyn_var_t tauh = (THC1 + (THC2 + THF*(v + THV1))*exp(-pow((v + THV2)/THD,2)))/T_ADJ;
    dyn_var_t tauh = (THC1 + (THC2 + THF*(v + THV1))*exp(-pow((v + THV2)/THD,2)))/getSharedMembers().Tadj;
    dyn_var_t pm = 0.5*dt/taum;
    dyn_var_t ph = 0.5*dt/tauh;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    h[i] = (2.0*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    }
#elif CHANNEL_KAs == KAs_WOLF_2005
    {
    // NOTE: Some models use m_inf and tau_m to estimate m
    //mtau = taum0  +  Cm * exp( - ((v-vthm)/vtcm)^2 )

    //left = alpha * exp( -(v-vth1-htaushift)/vtc1 )  : originally exp((v-vth1)/vtc1)
    //right = beta * exp( (v-vth2-htaushift)/vtc2 ) : originally exp(-(v-vth2)/vtc2)
    //htau = Ch  /  ( left + right )
    //dyn_var_t tau_m = 0.378 + 9.91 * exp(-pow((v + 34.3) / 30.1, 2)); //at 35^C
    dyn_var_t tau_m = 3.4 + 89.2 * exp(-pow((v + 34.3) / 30.1, 2));
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
    dyn_var_t h_a = AHC * exp(-(v + AHV) / AHD);
    dyn_var_t h_b = BHC * exp(-(v + BHV) / BHD);
    //dyn_var_t tau_h = 1097.4 / (h_a + h_b);
    dyn_var_t tau_h = 9876.6 / (h_a + h_b);
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    }
#else
    NOT IMPLEMENTED YET
#endif
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep h in [0, 1]
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }

#if CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
    g[i] = gbar[i]*m[i]*m[i]*h[i];
#elif CHANNEL_KAs == KAs_WOLF_2005
    g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelKAs::initialize(RNG& rng)
{
  pthread_once(&once_KAs, ChannelKAs::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels KAs (KAp) Param"
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
      if (gbar_values.size() != gbar_branchorders.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
          << "; gbar_branchorders.size = " << gbar_branchorders.size() << std::endl;
      }
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
#if CHANNEL_KAs == KAs_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
#elif CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
    m[i] = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    h[i] = 1.0/(1.0 + exp((v + IHV)/IHD));
    g[i] = gbar[i]*m[i]*m[i]*h[i];
#else
    NOT IMPLEMENTED YET
// m[i] = am / (am + bm); //steady-state value
// h[i] = ah / (ah + bh);
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelKAs::initialize_others() {}

ChannelKAs::~ChannelKAs() {}
