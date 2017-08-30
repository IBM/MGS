#include "CG_ChannelKAf_KChIP.h"
#include "ChannelKAf_KChIP.h"
#include "Lens.h"
#include "rndm.h"

#include <math.h>
#include <pthread.h>
#include <algorithm>

#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "SegmentDescriptor.h"

#define SMALL 1.0E-6
// unit conversion
#define uM2mM 1e-3
//
#define Cai_base 0.1  // [uM]



static pthread_once_t once_KAf = PTHREAD_ONCE_INIT;

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
// a_m  = AMC*(V - AMV)/( exp( (V - AMV)/AMD ) - 1.0 )
// b_m  = BMC * exp( (V - BMV)/BMD )
// a_h  = AHC * exp( (V - AHV)/AHD )
//
// This is an implementation of the fast component of A-type (KAf, KAt)
// potassium current
//
#if CHANNEL_KAf == KAf_TRAUB_1994
// The time constants in Traub's models is ~25 ms typically KAf ~ 30ms
#define Eleak -65.0  // mV
#define AMC -0.02
#define AMV (13.1 + Eleak)
#define AMD -10.0
#define BMC 0.0175
#define BMV (40.1 + Eleak)
#define BMD 10.0
#define AHC 0.0016
#define AHV (-13.0 + Eleak)
#define AHD -18
#define BHC 0.05
#define BHV (60.1 + Eleak)
#define BHD -5.0

#elif CHANNEL_KAf == KAf_KORNGREEN_SAKMANN_2000
// Korngreen - Sakmann (2000) Vm-gated K+ channels in PL5 neocortical young rat
// recorded in soma using nucleated outside-out patches
//          ... dendrite up to 430 um from soma using cell-attached recording
// NOTE: estimated patch surface area 440+/- 10 um^2
//             average series resistance 13.2+/-0.7 M.Ohm
//                    input resistance 2.7+/- 0.2 G.Ohm
//                    capacitance      2.2+/- 0.1 pF
#define IMV 10.0
#define IMD 19.0
#define IHV 76.0
#define IHD 10.0
#define TMC 0.34
#define TMF 0.92
#define TMV 81.0
#define TMD 59.0
#define THC 8.0
#define THF 49.0
#define THV 83.0
#define THD 23.0

#elif CHANNEL_KAf == KAf_MAHON_2000   
// Mahon 2000
// IKAf = g * m * h * (V-E)
#define VHALF_M -33.1                 
#define k_M -7.5                       
#define VHALF_H -70.4                 
#define k_H 7.6                      

#define tau_m 1.0 //ms                    
#define tau_h 25.0   // ms                 

#elif CHANNEL_KAf == KAf_EVANS_2012
//  Inactivation reference from  Tkatch - Surmeier (2000)
//     young adult rat (4-6 weeks postnatal) neostriatal spiny neuron
//     assume Kv4.2 subunits forming the channel
#define Eleak 0.0
#define AMC 1.5
#define AMV (4.0 + Eleak)
#define AMD -17
#define BMC 0.6
#define BMV (10.0 + Eleak)
#define BMD 9.0
#define AHC 0.105
#define AHV (-121.0 + Eleak)
#define AHD 22
#define BHC 0.065
#define BHV (-55.0 + Eleak)
#define BHD -11.0
#elif CHANNEL_KAf == KAf_WOLF_2005
//  Inactivation reference from  Tkatch - Surmeier (2000)
//     young adult rat (4-6 weeks postnatal) neostriatal spiny neuron
//     assume Kv4.2 subunits forming the channel
//    1. Tkatch et al. (2000) (V1/2: pg. 581, slope = Fig.3.B, tau: Fig.3C)
//  Activation reference from
//     1. Tkatch et al. (2000) (V1/2: pg. 581, slope corrected -17.7)
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -10.0
#define k_M -17.7
#define VHALF_H -75.6
#define k_H 10.0
#define LOOKUP_TAUM_LENGTH 11  // size of the below array
const dyn_var_t ChannelKAf_KChIP::_Vmrange_taum[] = {-40, -30, -20, -10, 0, 10,
                                                     20,  30,  40,  50,  60};
dyn_var_t ChannelKAf_KChIP::taumKAf[] = {1.8, 1.1, 1.0, 1.0, 0.9, 0.8,
                                         0.9, 0.9, 0.9, 0.8, 0.8};
std::vector<dyn_var_t> ChannelKAf_KChIP::Vmrange_taum;
#else
NOT IMPLEMENTED YET
#endif


// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKAf_KChIP::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
//#ifdef MICRODOMAIN_CALCIUM
//    //TUAN TODO: put _offset+i instead of 'i' here for MICRODOMAIN_CALCIUM
//    dyn_var_t cai = (*Cai)[i+_offset]; // [uM]
//    //std::cout << "Please update strategy for using Cacyto into modulating Kv4.2 channel" << std::endl;
//    //assert(0);
//#else
//    dyn_var_t cai = (*Cai)[i]; // [uM]
//#endif
//    float gbarAdj =  KChIP_Cav_on_conductance(cai);
    dyn_var_t cai;
    float vm_shift = 0;
    float gbarAdj = 1.0;
    float vm_slope_shift = 0;
#ifdef MICRODOMAIN_CALCIUM
    if (Cai){
      cai = (*Cai)[i+_offset]; // [uM]
      gbarAdj = KChIP_Cav_on_conductance(cai);
#define cads_max 30.0 
      vm_shift = -5 * std::min(1.0, cai/cads_max);
      vm_slope_shift = 5 * std::min(1.0, cai/cads_max);
    }
      //TUAN MODEL HACK
#endif

#if CHANNEL_KAf == KAf_TRAUB_1994
    {
      dyn_var_t am = AMC * vtrap((v - AMV), AMD);
      dyn_var_t bm = (BMC * (v - BMV)) / (exp((v - BMV) / BMD) - 1);
      dyn_var_t ah = AHC * exp((v - AHV) / AHD);
      dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
      // Traub Models do not have temperature dependence and hence Tadj is not
      // used
      dyn_var_t pm = 0.5 * dt * (am + bm);
      m[i] = (dt * am + m[i] * (1.0 - pm)) / (1.0 + pm);
      dyn_var_t ph = 0.5 * dt * (ah + bh);
      h[i] = (dt * ah + h[i] * (1.0 - ph)) / (1.0 + ph);
    }
#elif CHANNEL_KAf == KAf_KORNGREEN_SAKMANN_2000
    {
      dyn_var_t minf = 1.0 / (1.0 + exp(-(v + IMV) / IMD));
      // dyn_var_t taum = (TMC + TMF*exp(-pow((v + TMV)/TMD,2)))/T_ADJ;
      dyn_var_t taum =
          (TMC + TMF * exp(-pow((v + TMV) / TMD, 2))) / getSharedMembers().Tadj;
      dyn_var_t hinf = 1.0 / (1.0 + exp((v + IHV) / IHD));
      // dyn_var_t tauh = (THC + THF*exp(-pow((v + THV)/THD,2)))/T_ADJ;
      dyn_var_t tauh =
          (THC + THF * exp(-pow((v + THV) / THD, 2))) / getSharedMembers().Tadj;
      dyn_var_t pm = 0.5 * dt / taum;
      dyn_var_t ph = 0.5 * dt / tauh;
      // Rempe-Chopp 2006
      m[i] = (2.0 * pm * minf + m[i] * (1.0 - pm)) / (1.0 + pm);
      h[i] = (2.0 * ph * hinf + h[i] * (1.0 - ph)) / (1.0 + ph);
    }
#elif CHANNEL_KAf == KAf_MAHON_2000                                
                                                                   
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));       
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));       
    
    dyn_var_t pm = 0.5*dt*getSharedMembers().Tadj / tau_m;        
    dyn_var_t ph = 0.5*dt*getSharedMembers().Tadj / tau_h;        
    m[i] = (2.0*pm*m_inf + m[i]*(1.0 - pm))/(1.0 + pm);            
    h[i] = (2.0*ph*h_inf + h[i]*(1.0 - ph))/(1.0 + ph);            
#elif CHANNEL_KAf == KAf_EVANS_2012
    {
      dyn_var_t am = AMC / (1.0 + exp((v - AMV) / AMD));
      dyn_var_t bm = BMC / (1.0 + exp((v - BMV) / BMD));
      dyn_var_t ah = AHC / (1.0 + exp((v - AHV) / AHD));
      dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
      dyn_var_t pm = 0.5 * dt * (am + bm) * getSharedMembers().Tadj;
      m[i] = (dt * am + m[i] * (1.0 - pm)) / (1.0 + pm);
      dyn_var_t ph = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj;
      h[i] = (dt * ah + h[i] * (1.0 - ph)) / (1.0 + ph);
    }
#elif CHANNEL_KAf == KAf_WOLF_2005
    {
      // NOTE: Some models use m_inf and tau_m to estimate m
      std::vector<dyn_var_t>::iterator low =
          std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
      int index = low - Vmrange_taum.begin();
      // dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKAf[index] * 2);
      dyn_var_t taum;
      if (index == 0)
        taum = taumKAf[0];
      else
        taum = linear_interp(Vmrange_taum[index - 1], taumKAf[index - 1],
                             Vmrange_taum[index], taumKAf[index], v);
      dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);
      // const dyn_var_t tau_h = 4.67; // for 35^C
      const dyn_var_t tau_h = 14.0;
      dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);

      dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - vm_shift) / (k_M - vm_slope_shift)));

      dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
      h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    }

#else
    NOT IMPLEMENTED YET;
#endif
    {
      // trick to keep m in [0, 1]
      if (m[i] < 0.0) { m[i] = 0.0; }
      else if (m[i] > 1.0) { m[i] = 1.0; }
      // trick to keep h in [0, 1]
      if (h[i] < 0.0) { h[i] = 0.0; }
      else if (h[i] > 1.0) { h[i] = 1.0; }
    }

#if CHANNEL_KAf == KAf_TRAUB_1994
    g[i] = gbar[i] * m[i] * h[i];
#elif CHANNEL_KAf == KAf_KORNGREEN_SAKMANN_2000
    g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i] * h[i];
#elif CHANNEL_KAf == KAf_MAHON_2000
    g[i] = gbar[i] * m[i] * h[i];    
#elif CHANNEL_KAf == KAf_EVANS_2012
    g[i] = gbar[i] * m[i] * m[i] * h[i];
#elif CHANNEL_KAf == KAf_WOLF_2005
    g[i] = gbar[i] * m[i] * m[i] * h[i];
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]); // at time (t+dt/2)
  }
}

// GOAL: To meet second-order derivative, the gates is calculated to 
//     give the value at time (t0+dt/2) using data voltage v(t0)
//  NOTE: 
//    If steady-state formula is used, then the calculated value of gates
//            is at time (t0); but as steady-state, value at time (t0+dt/2) is the same
//    If non-steady-state formula (dy/dt = f(v)) is used, then 
//        once gate(t0) is calculated using v(t0)
//        we need to estimate gate(t0+dt/2)
//                  gate(t0+dt/2) = gate(t0) + f(v(t0)) * dt/2 
void ChannelKAf_KChIP::initialize(RNG& rng)
{
  pthread_once(&once_KAf, ChannelKAf_KChIP::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  if (not V)
  {
    std::cerr << typeid(*this).name()
              << " needs Voltage as input in ChanParam\n";
    assert(V);
  }
#if defined(SIMULATE_CACYTO)
  if (not Cai)
  {
    std::cerr << typeid(*this).name()
              << " needs Calcium as input in ChanParam to simulate Ca(domain)\n";
    //assert(Cai);
  }
#endif
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (Iion.size() != size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on "
                 "Channels KAf (KAt) Param"
              << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0)
    {
      unsigned int j;
      // NOTE: 'n' bins are splitted by (n-1) points
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
                  << "; gbar_dists.size = " << gbar_dists.size() << std::endl;
      }
      assert(gbar_values.size() - 1 == gbar_dists.size());
      for (j = 0; j < gbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j];
    }
    else if (gbar_branchorders.size() > 0)
    {
      unsigned int j;
      if (gbar_values.size() != gbar_branchorders.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
                  << "; gbar_branchorders.size = " << gbar_branchorders.size()
                  << std::endl;
      }
      assert(gbar_values.size() == gbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j = 0; j < gbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            gbar_branchorders[j])
          break;
      }
      if (j == gbar_branchorders.size() and
          gbar_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        gbar[i] = gbar_values[j - 1];
      }
      else if (j < gbar_values.size())
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    }
    else
    {
      gbar[i] = gbar_default;
    }
  }
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
//#ifdef MICRODOMAIN_CALCIUM
//    //TUAN TODO: put _offset+i instead of 'i' here for MICRODOMAIN_CALCIUM
//    dyn_var_t cai = (*Cai)[i+_offset]; // [uM]
//    //std::cout << "Please update strategy for using Cacyto into modulating Kv4.2 channel" << std::endl;
//    //assert(0);
//#else
//    dyn_var_t cai = (*Cai)[i]; // [uM]
//#endif
//    float gbarAdj = KChIP_Cav_on_conductance(cai);
    float gbarAdj = 1.0;
#ifdef MICRODOMAIN_CALCIUM
    dyn_var_t cai;
    if (Cai){
      cai = (*Cai)[i+_offset]; // [uM]
      gbarAdj = KChIP_Cav_on_conductance(cai);
    }
#endif

#if CHANNEL_KAf == KAf_TRAUB_1994
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    dyn_var_t bm = (BMC * (v - BMV)) / (exp((v - BMV) / BMD) - 1);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
    g[i] = gbar[i] * m[i] * h[i];
#elif CHANNEL_KAf == KAf_KORNGREEN_SAKMANN_2000
    m[i] = 1.0 / (1.0 + exp(-(v + IMV) / IMD));
    h[i] = 1.0 / (1.0 + exp((v + IHV) / IHD));
    g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i] * h[i];
#elif CHANNEL_KAf == KAf_MAHON_2000                  
    m[i] = 1.0 / (1 + exp(-(v - VHALF_M) / k_M));    
    h[i] = 1.0 / (1 + exp(-(v - VHALF_H) / k_H));    
    g[i] = gbar[i] * m[i] * h[i];                    
#elif CHANNEL_KAf == KAf_EVANS_2012
    dyn_var_t am = AMC / (1.0 + exp((v - AMV) / AMD));
    dyn_var_t bm = BMC / (1.0 + exp((v - BMV) / BMD));
    dyn_var_t ah = AHC / (1.0 + exp((v - AHV) / AHD));
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
    g[i] = gbar[i] * m[i] * h[i];
#elif CHANNEL_KAf == KAf_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * m[i] * h[i];
#else
    NOT IMPLEMENTED YET;
// m[i] = am / (am + bm);  // steady-state value
// h[i] = ah / (ah + bh);
#endif

    Iion[i] = g[i] * gbarAdj * (v - getSharedMembers().E_K[0]); // time (t0+dt/2)
  }
}

void ChannelKAf_KChIP::initialize_others()
{
#if CHANNEL_KAf == KAf_WOLF_2005
  std::vector<dyn_var_t> tmp(_Vmrange_taum, _Vmrange_taum + LOOKUP_TAUM_LENGTH);
  assert(sizeof(taumKAf) / sizeof(taumKAf[0]) == tmp.size());
  // Vmrange_taum.resize(tmp.size()-2);
  // for (unsigned long i = 1; i < tmp.size() - 1; i++)
  //  Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  Vmrange_taum = tmp;
#endif
}

ChannelKAf_KChIP::~ChannelKAf_KChIP() {}

#ifdef MICRODOMAIN_CALCIUM
void ChannelKAf_KChIP::setCalciumMicrodomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelKAf_KChIPInAttrPSet* CG_inAttrPset, CG_ChannelKAf_KChIPOutAttrPSet* CG_outAttrPset) 
{
  microdomainName = CG_inAttrPset->domainName;
  int idxFound = 0;
  while((*(getSharedMembers().tmp_microdomainNames))[idxFound] != microdomainName)
  {
    idxFound++;
  }
  _offset = idxFound * branchData->size;

}
#endif

