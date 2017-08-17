#include "Lens.h"
#include "ChannelNat.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

static pthread_once_t once_Nat = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "TTX-sensitive rapidly-activating, and
// rapidly-inactivating
//         Vm-gated Na^+ current, I_Nat (or I_Naf)".
#if CHANNEL_NAT == NAT_HODGKIN_HUXLEY_1952
// Gate: m^3 * h
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
#elif CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990
// Data from neostriatum neurons in guinea pig 200g
//    either sex (male/female)
//[Ogata, Tatebayashi, 1990]
//    Sodium current kinetics in isolated neostriatal neurons in adult guinea pig
//
#define VHALF_M -25
//#define VHALF_M -33  // TUAN - modified
#define k_M -10.0
#define VHALF_H -62
#define k_H 6
#define LOOKUP_TAUM_LENGTH 16  // size of the below array
const dyn_var_t ChannelNat::_Vmrange_taum[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                   -20,  -10, 0,   10,  20,  30,  40,  50};
dyn_var_t ChannelNat::taumNat[] = {0.3162, 0.3162, 0.3512, 0.4474, 0.5566, 0.3548, 0.2399, 0.1585,
                       0.1047, 0.0871, 0.0851, 0.0813, 0.0832, 0.0832, 0.0832, 0.0832};
//Below used in Evans et al. (2012)
//dyn_var_t ChannelNat::taumNat[] = {0.3162, 0.3162, 0.3162, 0.4074, 0.6166, 0.3548, 0.2399, 0.1585,
//                       0.1047, 0.0871, 0.0851, 0.0813, 0.0832, 0.0832, 0.0832, 0.0832};
#define LOOKUP_TAUH_LENGTH 16  // size of the below array
const dyn_var_t ChannelNat::_Vmrange_tauh[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                   -20,  -10, 0,   10,  20,  30,  40,  50};
dyn_var_t ChannelNat::tauhNat[] = {5.9196, 5.9196, 5.9197, 6.9103, 8.2985, 3.9111, 1.4907, 0.6596,
                       0.5101, 0.4267, 0.3673, 0.3370, 0.3204, 0.3177, 0.3151, 0.3142};
//Below used in Evans et al. (2012)
//dyn_var_t ChannelNat::tauhNat[] = {1.5,  1.5, 1.5,  1.5,  1.5,  1.5,  1.5136,  0.6761,
//                       0.5129, 0.4365, 0.3715, 0.3388, 0.2951, 0.2884, 0.2754, 0.2754};
std::vector<dyn_var_t> ChannelNat::Vmrange_taum;
std::vector<dyn_var_t> ChannelNat::Vmrange_tauh;
#elif CHANNEL_NAT == NAT_RUSH_RINZEL_1994
// Rush-Rinzel (1994) thalamic neuron
// adopted from HH-1952 data with
//  the kinetics has been adjusted to 35-degree C using Q10-factor=3
// NOTE:
// the channel model is almost fully inactivate (h~0.1 at -65mV)
// and it requires the cell to be hyperpolarized a certain amount of time
// to remove the inactivation
// NOTE: Use original HH-1952, but shift V-dependent upward along voltage-axis
//    hNa is replaced by a linear function of potassium inactivation hNa=0.85-n
//    to approximate sodium activation 'm'
//  with 'n' comes from KDR channel
#define Vmshift 10.3  //positive = shift to right-side
#define AMC 0.1
#define AMV (-35 + Vmshift)
#define AMD 10.0
#define BMC 4.0
#define BMV (-60 + Vmshift)
#define BMD 20.0
// This is 5/170 to account for the 170 in \tau_h
#define AHC 0.029411764705882
#define AHV -60.0
#define AHD 15.0
// This is 1/170 to account for the 170 in \tau_h
#define BHC 0.005882352941176
#define BHV -50.0
#define BHD 10.0
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
#define AHV (17.0+Eleak)
#define AHD -18.0
#define BHC 4.0
#define BHV (40.0+Eleak)
#define BHD -5.0
#elif CHANNEL_NAT == NAT_WANG_BUSZAKI_1996
// Wang, Buzsaki (1996) - Gamma oscillation by synaptic inhibition in hippocampal interneuron network model (paper has correction version)
// model after interneuron in hippocampal (based on Hodgkin-Huxley formula
// with the assumption fast-kinetic of 'm' gate, i.e. using m_infty)
// Ina = gNa * m_infty^3 * h (Vm-Ena)
// m_infty = a_m/(a_m + b_m)
// a_m = -0.1 * (Vm + 35) /(exp(-0.1*(Vm+35)) - 1.0)
// b_m = 4.0 * exp(-(Vm + 60)/18)
// dh/dt = Phi * (a_h * (1-h) - b_h * h)
// Phi = Q10^... = 5  --> temperature adjustment
// a_h = 0.07 * exp(-(Vm + 58)/20)
// b_h = 1.0/(exp(-0.1(Vm + 28)) + 1.0)
// ENa = 55 (mV)
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define Vshift 0  // [mV]
#define AMC -0.1        
#define AMV (-35.0+Vshift)
#define AMD -10          
#define BMC 4.0         
#define BMV (-60.0+Vshift) 
#define BMD -18.0         
#define AHC 0.07         
#define AHV (-58.0+Vshift)
#define AHD -20.0        
#define BHC 1.0         
#define BHV (-28.0+Vshift)
#define BHD -10.0        

// NOTE: Used by
//    1. Mahon et al. (2000) for MSN - striatum with Vshift = 7.0 (mV)

#elif CHANNEL_NAT == NAT_SCHWEIGHOFER_1999
// Gate: m_inf^3 * h
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

#elif CHANNEL_NAT == NAT_MAHON_2000                                                 
// Gate: m_inf^3 * h                                                                
//Reference from Wang Buzsaki (FSI neuron in neocortex/hippocampus)                 
//            but voltages shifted by 7mv,                                          
//m is substitute by its steady state value as activation variable m is assumed fast
// m_infty = a_m/(a_m + b_m)
// dh/dt = Phi * (a_h * (1-h) - b_h * h)
// a_m  = AMC*(V - AMV)/( exp( (V - AMV)/AMD  ) - 1.0  )                              
// b_m  = BMC * exp( (V - BMV)/BMD  )                                                
// a_h  = AHC * exp( (V - AHV)/AHD  )                                                
// b_h  = BHC / (exp( (V - BHV)/BHD  ) + 1.0)                                        
// NOTE: gNa = 1.20 nS/um^2 (equivalent to 120 mS/cm^2)                             
//   can be used with Q10 = 3                                                       
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define Vshift 7  // [mV]                                                           
#define AMC -0.1                                                                    
#define AMV (-35.0+Vshift)                                                          
#define AMD -10                                                                     
#define BMC 4.0                                                                     
#define BMV (-60.0+Vshift)                                                          
#define BMD -18.0                                                                   
#define AHC 0.07                                                                    
#define AHV (-58.0+Vshift)                                                          
#define AHD -20.0                                                                   
#define BHC 1.0                                                                     
#define BHV (-28.0+Vshift)                                                          
#define BHD -10.0                                                                   

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
// Inactivation from
//    1. Martina and Jonas (1997) - Table 1
// Activation from
//    1. Martina and Jonas (1997) - Table 1
//
// model is used for simulation NAc nucleus accumbens (medium-sized spiny MSN
// cell)
//    at 35.0 Celcius
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -23.9  //original Wolf
//#define VHALF_M -30.9
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
#else
  NOT IMPLEMENTED YET
#endif

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelNat::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2+dt)
//   of second-order accuracy at time (t+dt/2+dt) using trapezoidal rule
void ChannelNat::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); // at time (t+dt/2)
#if CHANNEL_NAT == NAT_HODGKIN_HUXLEY_1952
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
    m[i] = am/(am+bm); // m = m_inf assumption of instantaneous
    // see Rempe-Chomp (2006)
    //dyn_var_t pm = 0.5 * dt * (am + bm) * getSharedMembers().Tadj;
    //m[i] = (dt * am * getSharedMembers().Tadj + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj;
    h[i] = (dt * ah * getSharedMembers().Tadj + h[i] * (1.0 - ph)) / (1.0 + ph);
    }

#elif CHANNEL_NAT == NAT_MAHON_2000 || \
      CHANNEL_NAT == NAT_WANG_BUSZAKI_1996
    {                                                                           
    // NOTE: Some models use alpha_m and beta_m to estimate m                   
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);                                 
    dyn_var_t bm = BMC * exp((v - BMV) / BMD);                                  
    m[i] = am / (am + bm);  // steady-state value 
    dyn_var_t ah =  AHC * exp((v - AHV) / AHD);                                 
    dyn_var_t bh =  BHC / (1.0 + exp((v - BHV) / BHD));                         
    dyn_var_t ph = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj;              
    h[i] = (dt * ah * getSharedMembers().Tadj + h[i] * (1.0 - ph)) / (1.0 + ph);
    }

#elif CHANNEL_NAT == NAT_WOLF_2005 || \
    CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990
    {
    // NOTE: Some models use m_inf and tau_m to estimate m
    // tau_m in the lookup table
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    assert(index>=0);
    //assert(Vmrange_taum[index-1] <= v and v <= Vmrange_taum[index]);
    dyn_var_t taum;
    if (index == 0)
      taum = taumNat[0];
    else if (index < LOOKUP_TAUM_LENGTH)
     taum = linear_interp(Vmrange_taum[index-1], taumNat[index-1], 
        Vmrange_taum[index], taumNat[index], v);
    else //assume saturation in taum when Vm > max-value
     taum = taumNat[index-1];
    //-->tau_m[i] = taumNat[index];
    // NOTE: dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m[i] * 2);
    //dyn_var_t qm = dt * getSharedMembers().Tadj / (taumNat[index] * 2);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);
    /* no need to search as they both use the same Vmrange
     * IF NOT< make sure you add this code
    std::vector<dyn_var_t>::iterator low= std::lower_bound(Vmrange_tauh.begin(),
    Vmrange_tauh.end(), v);
    int index = low-Vmrange_tauh.begin();
    */
    dyn_var_t tauh;
    if (index == 0)
      tauh = tauhNat[0];
    else if (index < LOOKUP_TAUH_LENGTH)
      tauh = linear_interp(Vmrange_tauh[index-1], tauhNat[index-1], 
        Vmrange_tauh[index], tauhNat[index], v);
    else //assume saturation in taum when Vm > max-value
     tauh = tauhNat[index-1];
    //dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhNat[index] * 2);
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauh * 2);

    //dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - Vhalf_m_shift[i]) / k_M));
    //dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H - Vhalf_h_shift[i]) / k_H));
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
      
    }
#elif CHANNEL_NAT == NAT_HAY_2011 || \
		  CHANNEL_NAT == NAT_COLBERT_PAN_2002
    {
    //dyn_var_t am = AMC * vtrap(-(v - AMV - Vhalf_m_shift[i]), AMD);
    //dyn_var_t bm = BMC * vtrap(-(v - BMV - Vhalf_m_shift[i]), BMD);  //(v+BMV)/(exp((v+BMV)/BMD)-1)
    //dyn_var_t ah = AHC * vtrap(-(v - AHV - Vhalf_h_shift[i]), AHD);
    //dyn_var_t bh = BHC * vtrap(-(v - BHV - Vhalf_h_shift[i]), BHD);
    dyn_var_t am = AMC * vtrap(-(v - AMV - Vhalf_m_shift), AMD);
    dyn_var_t bm = BMC * vtrap(-(v - BMV - Vhalf_m_shift), BMD);  //(v+BMV)/(exp((v+BMV)/BMD)-1)
    dyn_var_t ah = AHC * vtrap(-(v - AHV - Vhalf_h_shift), AHD);
    dyn_var_t bh = BHC * vtrap(-(v - BHV - Vhalf_h_shift), BHD);
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
  //if (Vhalf_m_shift.size() !=size) Vhalf_m_shift.increaseSizeTo(size);
  //if (Vhalf_h_shift.size() !=size) Vhalf_h_shift.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
  //float Vhalf_m_default = Vhalf_m_shift[0];
  //float Vhalf_h_default = Vhalf_h_shift[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Nat Param"
      << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    //Vhalf_m_shift[i] = 0.0; //[mV]
    //Vhalf_h_shift[i] = 0.0; //[mV]
    //Vhalf_m_shift[i] = Vhalf_m_shift_default; //[mV]
    //Vhalf_h_shift[i] = Vhalf_h_shift_default; //[mV]
    //#if CHANNEL_NAT == NAT_COLBERT_PAN_2002
    //		//Vhalf_m init
    //		//NOTE: Shift to the left V1/2 for Nat in AIS region
    //#define DIST_START_AIS   30.0 //[um]
    //		if ((segmentDescriptor.getBranchType(branchData->key) == Branch::_AIS)
    //		// 	or 
    //		//	(		(segmentDescriptor.getBranchType(branchData->key) == Branch::_AXON)  and
    //	  //		 	(*dimensions)[i]->dist2soma >= DIST_START_AIS)
    //			)
    //		{
    //			//gbar[i] = gbar[i] * 1.50; // increase 3x
    //			//Vhalf_m_shift[i] = -15.0 ; //[mV]
    //			//Vhalf_h_shift[i] = -3.0 ; //[mV]
    //		}
    //#endif

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
#if CHANNEL_NAT == NAT_HODGKIN_HUXLEY_1952
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
#elif CHANNEL_NAT == NAT_MAHON_2000 || \
      CHANNEL_NAT == NAT_WANG_BUSZAKI_1996
    {                                                 
//v is at time (t0)
// so m and h is also at time t0
// however, as they are at steady-state, the value at time (t0+dt/2)
// does not change
      dyn_var_t am = AMC * vtrap((v - AMV), AMD);       
      dyn_var_t bm = BMC * exp((v - BMV) / BMD);        
      dyn_var_t ah = AHC * exp((v - AHV) / AHD);        
      dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
      m[i] = am / (am + bm);  // steady-state value     
      h[i] = ah / (ah + bh);                            
    }                                                 

#elif CHANNEL_NAT == NAT_HAY_2011 || \
    CHANNEL_NAT == NAT_COLBERT_PAN_2002
    {
      //dyn_var_t am = AMC * vtrap(-(v - AMV), AMD);
      //dyn_var_t am = AMC * vtrap(-(v - AMV), AMD);
      //dyn_var_t am = AMC * vtrap(-(v - AMV - Vhalf_m_shift[i]), AMD);
      //dyn_var_t bm = BMC * vtrap(-(v - BMV - Vhalf_m_shift[i]), BMD);  //(v+BMV)/(exp((v+BMV)/BMD)-1)
      //dyn_var_t ah = AHC * vtrap(-(v - AHV - Vhalf_h_shift[i]), AHD);
      //dyn_var_t bh = BHC * vtrap(-(v - BHV - Vhalf_h_shift[i]), BHD);
      dyn_var_t am = AMC * vtrap(-(v - AMV - Vhalf_m_shift), AMD);
      dyn_var_t bm = BMC * vtrap(-(v - BMV - Vhalf_m_shift), BMD);  //(v+BMV)/(exp((v+BMV)/BMD)-1)
      dyn_var_t ah = AHC * vtrap(-(v - AHV - Vhalf_h_shift), AHD);
      dyn_var_t bh = BHC * vtrap(-(v - BHV - Vhalf_h_shift), BHD);
      m[i] = am / (am + bm);  // steady-state value
      h[i] = ah / (ah + bh);
    }
#elif CHANNEL_NAT == NAT_WOLF_2005 || \
    CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
#else
    assert(0);
#endif

#if CHANNEL_NAT == NAT_TRAUB_1994
    g[i] = gbar[i] * m[i] *  m[i] * h[i];
#else 
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i]; // at time (t+dt/2) - 
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); //using 'v' at time 't'; but gate(t0+dt/2)
  }
}

void ChannelNat::initialize_others()
{
#if CHANNEL_NAT == NAT_WOLF_2005 || \
    CHANNEL_NAT == NAT_OGATA_TATEBAYASHI_1990
  {
    //NOTE: 
    //  0 <= i < size-1: _Vmrange_tauh[i] << Vm << _Vmrange_tauh[i+1]
    //  or 
    //  i = size-1: _Vmrange_tauh[i] << Vm 
    //  then :  taum[i]
    std::vector<dyn_var_t> tmp(_Vmrange_taum,
                               _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    //assert((sizeof(taumNat) / sizeof(taumNat[0])) == tmp.size());
		//Vmrange_taum.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_taum = tmp;
  }
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhNat) / sizeof(tauhNat[0]) == tmp.size());
		//Vmrange_tauh.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_tauh = tmp;
  }
#endif
}
  

ChannelNat::~ChannelNat() {}
