#include "Lens.h"
#include "ChannelNat.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

static pthread_once_t once_Nat = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "TTX-sensitive rapidly-activating, and
// rapidly-inactivating
//         Vm-gated Na^+ current, I_Nat (or I_Naf)".
#if CHANNEL_NAT == NAT_HODGKINHUXLEY_1952
// data measured from squid giant axon
//   at temperature = 6.3-degree celcius
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
// a_m  = AMC*(V - AMV)/( exp( (V - AMV)/AMD ) - 1.0 )
// b_m  = BMC * exp( (V - BMV)/BMD )
// a_h  = AHC * exp( (V - AHV)/AHD )
// b_h  = BHC / (exp( (V - BHV)/BHD ) + 1.0)
// NOTE: gNa = 1.20 nS/um^2 (equivalent to 120 mS/cm^2)
//   can be used with Q10 = 3
#define AMC 0.1
#define AMV -40.0
#define AMD 10.0
#define BMC 4.0
#define BMV -65.0
#define BMD 18.0
#define AHC 0.07
#define AHV -65.0
#define AHD 20.0
#define BHC 1.0
#define BHV -35.0
#define BHD 10.0
#elif CHANNEL_NAT == NAT_RUSH_RINZEL_1994
// Rush-Rinzel (1994) thalamic neuron
// adopted from HH-1952 data with
//  the kinetics has been adjusted to 35-degree C using Q10=3
// NOTE:
// the channel model is almost fully inactivate (h~0.1 at -65mV)
// and it requires the cell to be hyperpolarized a certain amount of time
// to remove the inactivation
// NOTE: Use original HH-1952, but shift V-dependent upward along voltage-axis
//    hNa is replaced by a linear function of potassium inactivation hNa=0.85-n
//    to approximate sodium activation 'm'
    assert(0);
#elif CHANNEL_NAT == NAT_TRAUB_1994
//Developed for Gamagenesis in interneurons
//All above conventions for a_m, a_h, b_h remain the same as above except b_m below
//b_m = (BMC * (V - BMV))/(exp((V-BMV)/BMD)-1)
#define Eleak -65.0 //mV
#define AMC -0.32
#define AMV (13.1+Eleak)
#define AMD -4.0
#define BMC 0.28
#define BMV (40.1+Eleak)
#define BMD 5.0
#define AHC 0.128
#define AHV (-17.0+Eleak)
#define AHD -18.0
#define BHC 4.0
#define BHV (-40.0+Eleak)
#define BHD -5.0
#elif CHANNEL_NAT == NAT_SCHWEIGHOFER_1999
// Developed for IO cell (inferior olive)
//     adapted from Rush-Rinzel (1994) thalamic neuron
//     to give long-lasting inactivation component
// data adjusted to 35-degree Celcius
#define AMC 0.1
#define AMV -41.0
#define AMD 10.0
#define BMC 9.0
#define BMV -66.0
#define BMD 20.0
// This is 5/170 to account for the 170 in \tau_h
#define AHC 0.029411764705882
#define AHV -60.0
#define AHD 15.0
// This is 1/170 to account for the 170 in \tau_h
#define BHC 0.005882352941176
#define BHV -50.0
#define BHD 10.0
//#endif
#elif CHANNEL_NAT == NAT_COLBERT_PAN_2002
// Kinetics data for Layer V5 pyramidal neuron
//       recorded at room tempt.(23-degree C)
//       from "Colbert-Pan (2002) - Nat. Neurosci (5)"
// This implementation uses kinetics of Nat from soma/den/IS
//  and it treats the Nat in AIS region by shifting V1/2 to the left 8mV
// This implementation uses kinetics of Nat from soma/den/IS
//    current,I_Nat".
#define AMC 0.182
#define AMV -40.0
#define AMD 6.0
#define BMC -0.124
#define BMV -40.0
#define BMD -6.0
#define AHC -0.015
#define AHV -66.0
#define AHD -6.0
#define BHC 0.015
#define BHV -66.0
#define BHD 6.0
#elif CHANNEL_NAT == NAT_WOLF_2005
// data from rat CA1 hippocampal pyramidal neuron
//     recorded at 22-24C and then mapped to 35C using Q10 = 3
// REF: Martina M, Jonas P (1997). "Functional differences in na+ channel gating
// between fast-spiking interneurons and principal neurons of rat hippocampus."
// J Phys,
// 505(3): 593-603.
//
// model is used for simulation NAc nucleus accumbens (medium-sized spiny MSN
// cell)
//    at 35.0 Celcius
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -23.9
#define k_M -11.8
#define VHALF_H -62.9
#define k_H 10.7
#define LOOKUP_TAUM_LENGTH 16  // size of the below array
const dyn_var_t ChannelNat::_Vmrange_taum[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                   -20,  -10, 0,   10,  20,  30,  40,  50};
// NOTE:
// if (-100+(-90))/2 >= Vm               : tau_m = taumNat[1st-element]
// if (-100+(-90))/2 < Vm < (-90+(-80))/2: tau_m = taumNat[2nd-element]
//...
dyn_var_t ChannelNat::taumNat[] = {0.06, 0.06, 0.07, 0.09, 0.11, 0.13, 0.20, 0.32,
                       0.16, 0.15, 0.12, 0.08, 0.06, 0.06, 0.06, 0.06};
#define LOOKUP_TAUH_LENGTH 16  // size of the below array
// dyn_var_t _Vmrange_tauh[] = _Vmrange_taum;
const dyn_var_t ChannelNat::_Vmrange_tauh[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                   -20,  -10, 0,   10,  20,  30,  40,  50};
dyn_var_t ChannelNat::tauhNat[] = {1.3,  1.3, 1.3,  1.3,  1.3,  1.3,  1.3,  1.3,
                       0.85, 0.5, 0.45, 0.32, 0.30, 0.28, 0.28, 0.28};
std::vector<dyn_var_t> ChannelNat::Vmrange_taum;
std::vector<dyn_var_t> ChannelNat::Vmrange_tauh;

#elif CHANNEL_NAT == NAT_HAY_2011
// Taken from Hay et al. (2011) "Models of Neocortical Layer 5b Pyramidal
// Cells..."
// which in turn references the work of Colbert et al. (2002) - NAT_COLBERT_PAN_2002
//   Q10=2.3 to map to 34-degree C
#define AMC 0.182
#define AMV -38.0
#define AMD 6.0
#define BMC -0.124
#define BMV -38.0
#define BMD -6.0
#define AHC -0.015
#define AHV -66.0
#define AHD -6.0
#define BHC 0.015
#define BHV -66.0
#define BHD 6.0
#endif

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelNat::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelNat::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_NAT == NAT_HODGKINHUXLEY_1952
    {
    // NOTE: Some models use alpha_m and beta_m to estimate m
    dyn_var_t am = AMC * vtrap(-(v - AMV), AMD);
    dyn_var_t bm = BMC * exp(-(v - BMV) / BMD);
    dyn_var_t ah = AHC * exp(-(v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp(-(v - BHV) / BHD));
    // see Rempe-Chopp (2006)
    dyn_var_t pm = 0.5 * dt * (am + bm) * getSharedMembers().Tadj;
    m[i] = (dt * am * getSharedMembers().Tadj + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj;
    h[i] = (dt * ah * getSharedMembers().Tadj + h[i] * (1.0 - ph)) / (1.0 + ph);
    }
#elif CHANNEL_NAT == NAT_TRAUB_1994
    {
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    // Traub Models do not have temperature dependence and hence Tadj is not used
    dyn_var_t pm = 0.5 * dt * (am + bm) ;
    m[i] = (dt * am  + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) ;
    h[i] = (dt * ah  + h[i] * (1.0 - ph)) / (1.0 + ph);
    }

#elif CHANNEL_NAT == NAT_SCHWEIGHOFER_1999
    {
    dyn_var_t am = AMC * vtrap(-(v - AMV), AMD);
    dyn_var_t bm = BMC * exp(-(v - BMV) / BMD);
    dyn_var_t ah = AHC * exp(-(v - AHV) / AHD);
    dyn_var_t bh = BHC * vtrap(-(v - BHV), BHD);
		m[i] = am/(am+bm); // assumption of instantaneous
    // see Rempe-Chomp (2006)
    //dyn_var_t pm = 0.5 * dt * (am + bm) * getSharedMembers().Tadj;
    //m[i] = (dt * am * getSharedMembers().Tadj + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj;
    h[i] = (dt * ah * getSharedMembers().Tadj + h[i] * (1.0 - ph)) / (1.0 + ph);
    }
#elif CHANNEL_NAT == NAT_WOLF_2005
    {
    // NOTE: Some models use m_inf and tau_m to estimate m
    // tau_m in the lookup table
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    //-->tau_m[i] = taumNat[index];
    // NOTE: dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m[i] * 2);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taumNat[index] * 2);
    /* no need to search as they both use the same Vmrange
     * IF NOT< make sure you add this code
    std::vector<dyn_var_t>::iterator low= std::lower_bound(Vmrange_tauh.begin(),
    Vmrange_tauh.end(), v);
    int index = low-Vmrange_tauh.begin();
    */
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhNat[index] * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
      
    }
#elif CHANNEL_NAT == NAT_HAY_2011 || \
		  CHANNEL_NAT == NAT_COLBERT_PAN_2002
    {
    dyn_var_t am = AMC * vtrap(-(v - AMV - Vhalf_m_shift[i]), AMD);
    dyn_var_t bm = BMC * vtrap(-(v - BMV - Vhalf_m_shift[i]), BMD);  //(v+BMV)/(exp((v+BMV)/BMD)-1)
    dyn_var_t ah = AHC * vtrap(-(v - AHV - Vhalf_h_shift[i]), AHD);
    dyn_var_t bh = BHC * vtrap(-(v - BHV - Vhalf_h_shift[i]), BHD);
    // see Rempe-Chomp (2006)
    dyn_var_t pm = 0.5 * dt * (am + bm) * getSharedMembers().Tadj;
    m[i] = (dt * am * getSharedMembers().Tadj + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj;
    h[i] = (dt * ah * getSharedMembers().Tadj + h[i] * (1.0 - ph)) / (1.0 + ph);

    }
#else
    assert(0);
#endif
    {//keep range [0..1]
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep h in [0, 1]
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
    }
   
#if CHANNEL_NAT == NAT_TRAUB_1994
    g[i] = gbar[i] * m[i] *  m[i] * h[i];
#else 
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i];
#endif

#ifdef WAIT_FOR_REST
		float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
		if (currentTime < NOGATING_TIME)
			g[i]= 0.0;
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);
  }
}

void ChannelNat::initialize(RNG& rng)
{
  pthread_once(&once_Nat, initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
	if (Vhalf_m_shift.size() !=size) Vhalf_m_shift.increaseSizeTo(size);
	if (Vhalf_h_shift.size() !=size) Vhalf_h_shift.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
	SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Nat Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
		Vhalf_m_shift[i] = 0.0; //[mV]
		Vhalf_h_shift[i] = 0.0; //[mV]
#if CHANNEL_NAT == NAT_COLBERT_PAN_2002
		//Vhalf_m init
		//NOTE: Shift to the left V1/2 for Nat in AIS region
#define DIST_START_AIS   30.0 //[um]
		if ((segmentDescriptor.getBranchType(branchData->key) == Branch::_AIS)
		// 	or 
		//	(		(segmentDescriptor.getBranchType(branchData->key) == Branch::_AXON)  and
	  //		 	(*dimensions)[i]->dist2soma >= DIST_START_AIS)
			)
		{
			//gbar[i] = gbar[i] * 1.50; // increase 3x
			Vhalf_m_shift[i] = -15.0 ; //[mV]
			Vhalf_h_shift[i] = -3.0 ; //[mV]
		}
#endif

		//gbar init
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
#if CHANNEL_NAT == NAT_HODGKINHUXLEY_1952
    {
    dyn_var_t am = AMC * vtrap(-(v - AMV), AMD);
    dyn_var_t bm = BMC * exp(-(v - BMV) / BMD);
    dyn_var_t ah = AHC * exp(-(v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp(-(v - BHV) / BHD));
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
    }
#elif CHANNEL_NAT == NAT_TRAUB_1994    
    {
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
    }
#elif CHANNEL_NAT == NAT_SCHWEIGHOFER_1999
    {
    dyn_var_t am = AMC * vtrap(-(v - AMV), AMD);
    dyn_var_t bm = BMC * exp(-(v - BMV) / BMD);
    dyn_var_t ah = AHC * exp(-(v - AHV) / AHD);
    dyn_var_t bh = BHC * vtrap(-(v - BHV), BHD);
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
    }
#elif CHANNEL_NAT == NAT_HAY_2011 || \
		  CHANNEL_NAT == NAT_COLBERT_PAN_2002
    //dyn_var_t am = AMC * vtrap(-(v - AMV), AMD);
    dyn_var_t am = AMC * vtrap(-(v - AMV - Vhalf_m_shift[i]), AMD);
    dyn_var_t bm = BMC * vtrap(-(v - BMV - Vhalf_m_shift[i]), BMD);  //(v+BMV)/(exp((v+BMV)/BMD)-1)
    dyn_var_t ah = AHC * vtrap(-(v - AHV - Vhalf_h_shift[i]), AHD);
    dyn_var_t bh = BHC * vtrap(-(v - BHV - Vhalf_h_shift[i]), BHD);
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
#elif CHANNEL_NAT == NAT_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
#else
	assert(0);
#endif

#if CHANNEL_NAT == NAT_TRAUB_1994
    g[i] = gbar[i] * m[i] *  m[i] * h[i];
#else 
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i];
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);
  }
}

void ChannelNat::initialize_others()
{
#if CHANNEL_NAT == NAT_WOLF_2005
  {
    std::vector<dyn_var_t> tmp(_Vmrange_taum,
                               _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    assert((sizeof(taumNat) / sizeof(taumNat[0])) == tmp.size());
		Vmrange_taum.resize(tmp.size()-2);
    for (unsigned long i = 1; i < tmp.size() - 1; i++)
      Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  }
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhNat) / sizeof(tauhNat[0]) == tmp.size());
		Vmrange_tauh.resize(tmp.size()-2);
    for (unsigned long i = 1; i < tmp.size() - 1; i++)
      Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  }
#endif
}
  

ChannelNat::~ChannelNat() {}
