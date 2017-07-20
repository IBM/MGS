#include "Lens.h"
#include "NMDAReceptor_Markov.h"
#include "CG_NMDAReceptor_Markov.h"
#include "rndm.h"
#include <iostream>
#include "math.h"
#include <limits>
#include "NTSMacros.h"
#include "NumberUtils.h"
#include "StringUtils.h"

#define SMALL 1.0E-6
#define decimal_places 6
#define fieldDelimiter "\t"
// dynamics of neurotransmitter
#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
#define NEUROTRANSMITTER      \
  (getSharedMembers().NTmax / \
   (1.0 + exp(-(*Vpre - getSharedMembers().Vp) / getSharedMembers().Kp)))

#elif SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT 
// NOTE: (should be updated in SynapticCleft nodetype)
#define NEUROTRANSMITTER      *Glut

#endif

// take into account the effect of temperature making the change faster or
// slower
#define DT (*(getSharedMembers().deltaT))
#define Tscale (*(getSharedMembers().deltaT) * (getSharedMembers().Tadj))
#define KETAMINE (*(getSharedMembers().Ketamine))
#define GLYCINE (*(getSharedMembers().Glycine))

#define W w
#define pOn (getSharedMembers().plasticityOn)
#define pStart (getSharedMembers().plasticityStartAt)
#define pStop (getSharedMembers().plasticityStopAt)

#if ! defined(SIMULATE_CACYTO)
//NOTE: TAU = time-constant for learning
//       i.e. Ca2+-dependent learning rate is 1/TAU
  #define Cai_base  0.1 // [uM]
#endif

//Different modeling of Mg2+ block
#if RECEPTOR_NMDA == NMDAR_BEHABADI_2012
// Mg2+ block from "Behabadi BF, Polsky A, Jadi M, Schiller J, Mel BW (2012)
// Location-Dependent Excitatory Synaptic Interactions in Pyramidal Neuron
// Dendrites. PLoS Comput Biol 8(7): e1002599. doi:10.1371/journal.pcbi.1002599"
// Mg2+ block as instantaneous function of Vm-post
//   NOTE: formula in the paper
//#define MGBLOCK (1.0 / (1.0 + exp(- (*Vpost)[indexPost] + 12.0) / 10.0))
//   NOTE: formula in senselab website
#define MGBLOCK (1.0 / (1.0 + 0.3 * exp(-0.1 * (*Vpost)[indexPost])))

#elif RECEPTOR_NMDA == NMDAR_BEHABADI_2012_MODIFIED
//#define MGBLOCK (1.0 / (1.0 + 0.3 * exp(-0.1136 * (*Vpost)[indexPost])))
#define MGBLOCK (1.0 / (1.0 + 0.3 * exp(-0.1316 * (*Vpost)[indexPost])))

#elif RECEPTOR_NMDA == NMDAR_JADI_2012
// Mg2+ block from "Jadi M, Polsky A, Schiller J, Mel BW (2012)
// Location-Dependent Effects of Inhibition on Local Spiking in Pyramidal Neuron
// Dendrites. PLoS Comput Biol 8(6): e1002550. doi:10.1371/journal.pcbi.1002550"
#define Kp_V  12.5 // [mV] the steepness of voltage dependency
#define MGBLOCK (1.0 / (1.0 + exp(-((*Vpost)[indexPost] + 7.0) / Kp_V)))

#elif RECEPTOR_NMDA == NMDAR_JAHR_STEVENS_1990
#define Kp_Mgion 3.57 // [mM] the steepness of voltage dependency
#define MGBLOCK                                 \
  (1.0 / (1.0 +                                 \
          exp(-0.062 * ((*Vpost)[indexPost])) * \
              (*(getSharedMembers().Mg_EC)) / Kp_Mgion))
//#define MGBLOCK 1.0/(1.0 +
// exp(-0.122*((*Vpost)[indexPost]))*(*(getSharedMembers().Mg_EC))/3.57)
////Adjusted sigmoid to not get calcium transients at -60mV

#elif RECEPTOR_NMDA == NMDAR_POINTPROCESS
// No Mg2+ blocks
#define MGBLOCK  1

#endif


// Destexhe-Mainen-Sejnowski (1994)
// (1.3.2)
// The gating of NMDAR is modeled using
//  NOTE: Here there is 1 desensitized state
//                     1         2             3
//   NMDAR        + R <=> R_Kai <=>  R(Kai)_2 <=>[alpha][beta] O
//                                     /\
//                                     ||  4
//                                     \/
//                                    RD(NMDA)_2
//   1. <=>[Rb* NT][Ru]
//   2. <=>[Rb* NT][Ru]
//   3. <=>[alpha][beta]
//   4. <=>[Rd][Rr]
// then given fO (fraction of NMDAR in Open state)
//    dfO/dt = alpha * fC2 - beta * fO
#define Ro (getSharedMembers().alpha)
#define Rc (getSharedMembers().beta)
#define Rb (getSharedMembers().Rb)
#define cRb (getSharedMembers().cRb)
#define Ru (getSharedMembers().Ru)
#define Rr (getSharedMembers().Rr)
#define Rd (getSharedMembers().Rd)

void NMDAReceptor_Markov::initializeNMDA(RNG& rng) 
{
#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
  assert(Vpre);
#endif
  assert(Vpost);
#if  defined(SIMULATE_CACYTO)
  assert(Ca_IC);
#endif
  assert(getSharedMembers().T != 0 && getSharedMembers().Ca_EC != 0 &&
         getSharedMembers().Mg_EC != 0);

  if (KETAMINE == 0)
  {
    KETAMINE = 0;
  }
  if (GLYCINE == 0)
  {
    GLYCINE = 0;
  }

  g = w * gbar * MGBLOCK * fO * (1 - KETAMINE);

  buffer = 0;
  gbar0 = gbar;

  if (pOn)
  {
    if (pOn == 1)
    {  // Graupner & Brunel 2012 PNAS
      tp = getSharedMembers().theta_p;
    }
    else if (pOn == 2)
    {
      tp = 0.55;
    }
  }

  #ifdef DEBUG_CHAN
    std::ostringstream os;
    std::ostringstream os2;
    std::string fileName = "NMDAR_Markov.dat";
    os << fileName << StringUtils::random_string(3) << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << ": States";
    os2 << "# time, C0, C1, C2, D, O" ;
    (*outFile) << "\n";
    (*outFile) << os2.str() << "\n";
  #endif
}

void NMDAReceptor_Markov::updateNMDA(RNG& rng) 
{
  // Calculate receptor conductance
  fO = Tscale * (Ro * fC2 - Rc * fO) + fO;
  fC2 = Tscale * ((Rb * NEUROTRANSMITTER + cRb) * fC1 + Rc * fO + Rr * fD -
              (Ro + Ru + Rd) * fC2) +
        fC2;
  fC1 = Tscale * (Rb * NEUROTRANSMITTER * fC0 + Ru * fC2  -
              (Rb * NEUROTRANSMITTER + Ru ) * fC1) +
        fC1;
  fD = Tscale * (Rd * fC2 - Rr * fD) + fD;

  if (fO < 0.0) { fO = 0.0; }
  else if (fO > 1.0) { fO = 1.0; }
  if (fC2 < 0.0) { fC2 = 0.0; }
  else if (fC2 > 1.0) { fC2 = 1.0; }
  if (fC1 < 0.0) { fC1 = 0.0; }
  else if (fC1 > 1.0) { fC1 = 1.0; }
  if (fD < 0.0) { fD = 0.0; }
  else if (fD > 1.0) { fD = 1.0; }

  fC0 = 1.0 - (fC1 + fC2 + fD + fO);
  if (fC0 < 0.0) { fC0 = 0.0; }
  else if (fC0 > 1.0) { fC0 = 1.0; }


  //TODO: TUAN incorporate the effect of Glycine into gating dynamics
  g = gbar * MGBLOCK * fO * (1 - KETAMINE) ;
//  if (getSimulation().getIteration().(*(getSharedMembers().deltaT))

#if ! defined(SIMULATE_CACYTO)
    dyn_var_t cai = Cai_base;
#else
#ifdef MICRODOMAIN_CALCIUM
    dyn_var_t cai = (*Ca_IC)[indexPost+_offset]; // [uM]
#else
    dyn_var_t cai = (*Ca_IC)[indexPost];
#endif

#endif

  // Updates the channel reversal potential
  // RT/(zCa*F) * ln(Cao/Cai)
  //E_Ca = (R_zCaF * *(getSharedMembers().T) *
  //        log(*(getSharedMembers().Ca_EC) / cai));

  dyn_var_t gCa = g;
  if (pOn == 1)
  {
    gCa = g / 10;
  }
  else if (pOn == 2)
  {
    gCa = g / 20;
  }

  //I_Ca = gCa * ((*Vpost)[indexPost] - E_Ca); // [pA/um^2]
  I_Ca = gCa * ((*Vpost)[indexPost] - getSharedMembers().E); // [pA/um^2]
  #ifdef DEBUG_CHAN
    (*outFile) << float(getSimulation().getIteration()) * DT;
    (*outFile) << std::fixed << fieldDelimiter << fC0;
    (*outFile) << std::fixed << fieldDelimiter << fC1;
    (*outFile) << std::fixed << fieldDelimiter << fC2;
    (*outFile) << std::fixed << fieldDelimiter << fD;
    (*outFile) << std::fixed << fieldDelimiter << fO;
    (*outFile) << "\n";
  #endif
}

void NMDAReceptor_Markov::updateNMDADepPlasticity(RNG& rng) 
{
#if ! defined(SIMULATE_CACYTO)
    dyn_var_t cai = Cai_base;
#else
#ifdef MICRODOMAIN_CALCIUM
    dyn_var_t cai = (*Ca_IC)[indexPost+_offset]; // [uM]
#else
    dyn_var_t cai = (*Ca_IC)[indexPost];
#endif
#endif
  if (pOn)
  {
    if ((getSimulation().getIteration() * DT) > pStart &&
        (getSimulation().getIteration() * DT) < pStop)
    {
      if (pOn == 1)
      {  // Graupner & Brunel 2012 PNAS
        dyn_var_t dw = (-w * (1.0 - w) * (getSharedMembers().w_th - w) +
                        getSharedMembers().gamma_p * (1.0 - w) *
                            ((dyn_var_t)((cai -
                                          getSharedMembers().theta_p) >= 0)) -
                        getSharedMembers().gamma_d * w *
                            ((dyn_var_t)((cai -
                                          getSharedMembers().theta_d) >= 0))) /
                       getSharedMembers().tau;
        w = w + DT * dw;

        if (getSharedMembers().deltaNMDAR)
        {  // Metaplasticity
          dyn_var_t dBuffer;

          if (dw > 0)
          {
            dBuffer = -buffer + dw;
          }
          else
          {
            dBuffer = -buffer;
          }

          buffer = buffer + dBuffer * DT;

          dyn_var_t dgbar = (gbar0 - gbar) / getSharedMembers().tauBuffer +
                            getSharedMembers().alphaBuffer * buffer;
          gbar = gbar + dgbar * DT;
        }
      }
      else if (pOn == 2)
      {  // Shouval & Bear & Cooper 2002 PNAS
  #define TAU (100.0 / (100.0 / 0.001 + pow(cai, 3)) + 1000.0)
  #define CAFUN                                       \
  	(0.25 + sigmoid(cai - 0.55, 80.0) - \
  	 0.25 * sigmoid(cai - 0.35, 80.0))
        w = w + (1.0 / TAU) * (CAFUN - w);
      }
    }
  }
}

void NMDAReceptor_Markov::setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_NMDAReceptor_MarkovOutAttrPSet* CG_outAttrPset) 
{
  indexPost = CG_inAttrPset->idx;
  if (indexPrePost.size() % 2)
  {//it means that PreSynapticPoint is being used
#ifdef KEEP_PAIR_PRE_POST
  indexPrePost.push_back(&indexPost);
#endif
  }
}

void NMDAReceptor_Markov::setPrePostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_NMDAReceptor_MarkovOutAttrPSet* CG_outAttrPset) 
{
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[0]);
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[1]);
}

NMDAReceptor_Markov::~NMDAReceptor_Markov() 
{
  #ifdef DEBUG_CHAN
    if (outFile) 
    {
      outFile->close();
    }
    delete outFile;
  #endif
}
#ifdef DEBUG_CHAN
NMDAReceptor_Markov::NMDAReceptor_Markov() 
   : CG_NMDAReceptor_Markov(), outFile(0)
{
}
#endif

#ifdef MICRODOMAIN_CALCIUM
void NMDAReceptor_Markov::setCalciumMicrodomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_NMDAReceptor_MarkovOutAttrPSet* CG_outAttrPset) 
{
  microdomainName = CG_inAttrPset->domainName;
  int idxFound = 0;
  while((*(getSharedMembers().tmp_microdomainNames))[idxFound] != microdomainName)
  {
    idxFound++;
  }
  //_offset = idxFound * branchData->size;
  //NOTE: Receptors always reside on post-side (this does not necessary means
  //   post-synaptic side. It is just the post-side in the pre-post touch)
  _offset = idxFound * branchDataPrePost[1]->size;
}
#endif

