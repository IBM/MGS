#include "CG_ChannelKAs.h"
#include "ChannelKAs.h"
#include "Lens.h"
#include "rndm.h"

#include "GlobalNTSConfig.h"
#include "SegmentDescriptor.h"
#include "Branch.h"
#include <pthread.h>

#define SMALL 1.0E-6
#define decimal_places 6     
#define fieldDelimiter "\t"  

static pthread_once_t once_KAs = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelKAs::_segmentDescriptor;
#endif

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

#elif CHANNEL_KAs == KAs_MAHON_2000
// Model adopted for MSN striatum
// model parameters from mahon 2000 Table 1
#define VHALF_M -25.6
#define k_M -13.3
#define VHALF_H -78.8
#define k_H 10.4

#define tau_M 131.4
#define VHALF_TAUM -37.4
#define k_TAUM 27.3

#define VHALF_TAUH -38.2
#define k_TAUH 28

#elif CHANNEL_KAs == KAs_WOLF_2005
//  Inactivation from young adult (3-4 weeks) Sprague-Dawley rat
//      - attributed to Kv1.2 subunits
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
#define AHV -90.96
#define AHD -29.01
#define BHC 1.0
#define BHV -90.96
#define BHD 100
#elif CHANNEL_KAs == KAs_EVANS_2012
// Also use Kv1.2 data from Shen et al. (2004)
//#define scale_tau_m  2.5
//#define scale_tau_h 2.5
#define scale_tau_m 1
#define scale_tau_h 1
#define VHALF_M -27.0
#define k_M -16.0
#define VHALF_H -33.5
#define k_H 21.5
#define frac_inact 1.0 //0.996  // 'a' term

#define AMC 0.250 // [1/ms]
#define AMV 50.0  // [mV]
#define AMD -20.0  // [mV]
#define BMC 0.05
#define BMV -90.0
#define BMD 35.0  //[mV]

#define AHC 0.0025
#define AHV -95
#define AHD 16
#define BHC 0.002
#define BHV 50.0
#define BHD -70
#else
NOT IMPLEMENTED YET
#endif

#ifndef scale_tau_m
#define scale_tau_m 1.0
#endif
#ifndef scale_tau_h
#define scale_tau_h 1.0 
#endif

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKAs::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
#if defined(WRITE_GATES)                                                  
  bool is_write = false;
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    float currentTime = float(getSimulation().getIteration()) * dt + dt/2;       
    if (currentTime >= _prevTime + IO_INTERVAL)                           
    {                                                                     
      (*outFile) << std::endl;                                            
      (*outFile) <<  currentTime;                                         
      _prevTime = currentTime;                                            
      is_write = true;
    }
  }
#endif
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
    {
      dyn_var_t minf = 1.0 / (1.0 + exp(-(v + IMV) / IMD));
      dyn_var_t taum;
      if (v < -60.0)
      {
        // taum = (TMC + TMF1*exp(TMD1*(v + TMV)))/T_ADJ;
        taum = (TMC + TMF1 * exp(TMD1 * (v + TMV))) / getSharedMembers().Tadj;
      }
      else
      {
        // taum = (TMC + TMF2*exp(TMD2*(v + TMV)))/T_ADJ;
        taum = (TMC + TMF2 * exp(TMD2 * (v + TMV))) / getSharedMembers().Tadj;
      }
      dyn_var_t hinf = 1.0 / (1.0 + exp((v + IHV) / IHD));
      // dyn_var_t tauh = (THC1 + (THC2 + THF*(v + THV1))*exp(-pow((v +
      // THV2)/THD,2)))/T_ADJ;
      dyn_var_t tauh =
          (THC1 + (THC2 + THF * (v + THV1)) * exp(-pow((v + THV2) / THD, 2))) /
          getSharedMembers().Tadj;
      dyn_var_t pm = 0.5 * dt / taum;
      dyn_var_t ph = 0.5 * dt / tauh;
      m[i] = (2.0 * pm * minf + m[i] * (1.0 - pm)) / (1.0 + pm);
      h[i] = (2.0 * ph * hinf + h[i] * (1.0 - ph)) / (1.0 + ph);
    }
#elif CHANNEL_KAs == KAs_MAHON_2000
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    dyn_var_t tau_m = (tau_M / (exp(-(v - VHALF_TAUM) / k_TAUM) +
                                exp((v - VHALF_TAUM) / k_TAUM))) /
                      getSharedMembers().Tadj;
    dyn_var_t tau_h = (1790 +
                       2930 * exp(-pow((v - VHALF_TAUH) / k_TAUH, 2)) *
                           ((v - VHALF_TAUH) / k_TAUH)) /
                      getSharedMembers().Tadj;
    dyn_var_t pm = 0.5 * dt / tau_m;
    dyn_var_t ph = 0.5 * dt / tau_h;
    m[i] = (2.0 * pm * m_inf + m[i] * (1.0 - pm)) / (1.0 + pm);
    h[i] = (2.0 * ph * h_inf + h[i] * (1.0 - ph)) / (1.0 + ph);

#elif CHANNEL_KAs == KAs_WOLF_2005
    {
      // NOTE: Some models use m_inf and tau_m to estimate m
      // mtau = taum0  +  Cm * exp( - ((v-vthm)/vtcm)^2 )

      // left = alpha * exp( -(v-vth1-htaushift)/vtc1 )  : originally
      // exp((v-vth1)/vtc1)
      // right = beta * exp( (v-vth2-htaushift)/vtc2 ) : originally
      // exp(-(v-vth2)/vtc2)
      // htau = Ch  /  ( left + right )
      // dyn_var_t tau_m = 0.378 + 9.91 * exp(-pow((v + 34.3) / 30.1, 2)); //at
      // 35^C
      dyn_var_t tau_m = 3.4 + 89.2 * exp(-pow((v + 34.3) / 30.1, 2)); // at 15^Celcius
      dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);

      dyn_var_t h_a = AHC * exp((v - AHV) / AHD);
      dyn_var_t h_b = BHC * exp((v - BHV) / BHD);
      // dyn_var_t tau_h = 1097.4 / (h_a + h_b); // at 35^Celcius
      dyn_var_t tau_h = 9876.6 / (h_a + h_b); // at 15^Celcius
      dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);

      dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
      dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
      h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    }
#elif CHANNEL_KAs == KAs_EVANS_2012
    {
      dyn_var_t am = AMC / (1 + exp((v - AMV) / AMD));
      dyn_var_t bm = BMC / (1 + exp((v - BMV) / BMD));
      dyn_var_t ah = AHC / (1 + exp((v - AHV) / AHD));
      dyn_var_t bh = BHC / (1 + exp((v - BHV) / BHD));
      dyn_var_t m_inf = am / (am+bm);
      dyn_var_t h_inf = ah / (ah+bh);
      //dyn_var_t tau_m = scale_tau_m * 1.0 / (am + bm);
      //dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
      dyn_var_t qm = 0.5 * dt * (am + bm) * getSharedMembers().Tadj / scale_tau_m;
      dyn_var_t qh = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj / scale_tau_h;
      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
      h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    }
#else
    NOT IMPLEMENTED YET;
#endif
    {
      // trick to keep m in [0, 1]
      if (m[i] < 0.0)
      {
        m[i] = 0.0;
      }
      else if (m[i] > 1.0)
      {
        m[i] = 1.0;
      }
      // trick to keep h in [0, 1]
      if (h[i] < 0.0)
      {
        h[i] = 0.0;
      }
      else if (h[i] > 1.0)
      {
        h[i] = 1.0;
      }
    }

#if CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
    g[i] = gbar[i] * m[i] * m[i] * h[i];
#elif CHANNEL_KAs == KAs_MAHON_2000
    g[i] = gbar[i] * m[i] * h[i];
#elif CHANNEL_KAs == KAs_WOLF_2005
    g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
#elif CHANNEL_KAs == KAs_EVANS_2012
    g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time (t+dt/2)
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
      (*outFile) << std::fixed << fieldDelimiter << h[i];                 
    }                                                                     
#endif                                                                    
  }
}

// GOAL: To meet second-order derivative, the gates is calculated to
//     give the value at time (t0+dt/2) using data voltage v(t0)
//  NOTE:
//    If steady-state formula is used, then the calculated value of gates
//            is at time (t0); but as steady-state, value at time (t0+dt/2) is
//            the same
//    If non-steady-state formula (dy/dt = f(v)) is used, then
//        once gate(t0) is calculated using v(t0)
//        we need to estimate gate(t0+dt/2)
//                  gate(t0+dt/2) = gate(t0) + f(v(t0)) * dt/2
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
  if (Iion.size() != size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on "
                 "Channels KAs (KAp) Param"
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

  //NOTE: 
  //scale_factor if NOT defined get the value of 1.0
  // if defined; dont use a closed to zero value, i.e. > SMALL
  if (scale_factor < SMALL)
  {
    scale_factor = 1.0;
  }
  else{
    for (unsigned i = 0; i < size; ++i)
    {
      gbar[i] *= scale_factor;
    }
  }
#if defined(WRITE_GATES)                                      
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    std::ostringstream os;                                    
    std::string fileName = "gates_KAs.txt";                       
    os << fileName << getSimulation().getRank();              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);                       
    (*outFile) << "#Time" << fieldDelimiter << "gates: m, h [, m,h]*"; 
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
    m[i] = 1.0 / (1.0 + exp(-(v + IMV) / IMD));
    h[i] = 1.0 / (1.0 + exp((v + IHV) / IHD));
    g[i] = gbar[i] * m[i] * m[i] * h[i];
#elif CHANNEL_KAs == KAs_MAHON_2000
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * h[i];

#elif CHANNEL_KAs == KAs_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
#elif CHANNEL_KAs == KAs_EVANS_2012
    {
      dyn_var_t am = AMC / (1 + exp((v - AMV) / AMD));
      dyn_var_t bm = BMC / (1 + exp((v - BMV) / BMD));
      dyn_var_t ah = AHC / (1 + exp((v - AHV) / AHD));
      dyn_var_t bh = BHC / (1 + exp((v - BHV) / BHD));
      m[i] = am / (am+bm);
      h[i] = ah / (ah+bh);
      g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
    }
#else
    NOT IMPLEMENTED YET;
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time t0+dt/2
#if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
      (*outFile) << std::fixed << fieldDelimiter << h[i];       
    }
#endif                                                        
  }
}

void ChannelKAs::initialize_others() {}

ChannelKAs::~ChannelKAs() 
{
#if defined(WRITE_GATES)            
  if (outFile) outFile->close();    
#endif                              
}
