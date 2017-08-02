// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "NMDAReceptor.h"
#include "CG_NMDAReceptor.h"
#include "rndm.h"
#include <iostream>
#include "math.h"
#include <limits>
#include "NTSMacros.h"
#include "NumberUtils.h"

// Destexhe-Mainen-Sejnowski (1994)
// The Glutamate neurotransmitter concentration, i.e. [NT]
//  which is assumed to be an instantaneous function of Vm
//  [NT] = NTmax / (1 + exp (-(Vm - Vp)/Kp))
//
// The gating of NMDAR is modeled using
//   C <==>[alpha * [NT]][beta] S -->[MGBLOCK] O
// then given r = fO (fraction of AMPAR in Open state)
//    dr/dt = alpha * NT * (1-r) - beta * r
#define ALPHA (getSharedMembers().alpha)
#define BETA (getSharedMembers().beta)

#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
#define NEUROTRANSMITTER      \
  (getSharedMembers().NTmax / \
   (1.0 + exp(-(*Vpre - getSharedMembers().Vp) / getSharedMembers().Kp)))

#elif SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT 
#define NEUROTRANSMITTER      *Glut

#endif

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

void NMDAReceptor::initializeNMDA(RNG& rng)
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

  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  r = ALPHANEUROTRANSMITTER / (BETA + ALPHANEUROTRANSMITTER);
  g = w * gbar * MGBLOCK * r * (1 - KETAMINE);

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
}

void NMDAReceptor::updateNMDA(RNG& rng)
{
  // Calculate receptor conductance
  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  dyn_var_t A = Tscale * (BETA + ALPHANEUROTRANSMITTER) / 2.0;
  r = (Tscale * ALPHANEUROTRANSMITTER + r * (1.0 - A)) / (1.0 + A);

	//TODO: TUAN incorporate the effect of Glycine into gating dynamics
  g = gbar * MGBLOCK * r * (1 - KETAMINE) ;
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
  E_Ca = (R_zCaF * *(getSharedMembers().T) *
          log(*(getSharedMembers().Ca_EC) / cai));

  dyn_var_t gCa = g;
  if (pOn == 1)
  {
    gCa = g / 10;
  }
  else if (pOn == 2)
  {
    gCa = g / 20;
  }

  I_Ca = gCa * ((*Vpost)[indexPost] - E_Ca); // [pA/um^2]
}

void NMDAReceptor::updateNMDADepPlasticity(RNG& rng)
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

void NMDAReceptor::setPostIndex(const String& CG_direction,
                                const String& CG_component,
                                NodeDescriptor* CG_node, Edge* CG_edge,
                                VariableDescriptor* CG_variable,
                                Constant* CG_constant,
                                CG_NMDAReceptorInAttrPSet* CG_inAttrPset,
                                CG_NMDAReceptorOutAttrPSet* CG_outAttrPset)
{
  indexPost = CG_inAttrPset->idx;
  if (indexPrePost.size() % 2)
  {//it means that PreSynapticPoint is being used
#ifdef KEEP_PAIR_PRE_POST
  indexPrePost.push_back(&indexPost);
#endif
  }
}


NMDAReceptor::~NMDAReceptor() {}
void NMDAReceptor::setPrePostIndex(const String& CG_direction,
                                const String& CG_component,
                                NodeDescriptor* CG_node, Edge* CG_edge,
                                VariableDescriptor* CG_variable,
                                Constant* CG_constant,
                                CG_NMDAReceptorInAttrPSet* CG_inAttrPset,
                                CG_NMDAReceptorOutAttrPSet* CG_outAttrPset)
{
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[0]);
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[1]);
}


#ifdef MICRODOMAIN_CALCIUM
void NMDAReceptor::setCalciumMicrodomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptorInAttrPSet* CG_inAttrPset, CG_NMDAReceptorOutAttrPSet* CG_outAttrPset) 
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

