// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "AMPAReceptor.h"
#include "CG_AMPAReceptor.h"
#include "rndm.h"
#include <iostream>
#include "NodeProxyBase.h"

// Destexhe-Mainen-Sejnowski (1994)
// The Glutamate neurotransmitter concentration, i.e. [NT]
//  which is assumed to be an instantaneous function of Vm
//  [NT] = NTmax / (1 + exp (-(Vm - Vp)/Kp))
//
// The gating of AMPAR is modeled using
//   C <==>[alpha * [NT]][beta] O
// then given r = fO (fraction of AMPAR in Open state)
//    dr/dt = alpha * NT * (1-r) - beta * r

#define ALPHA (getSharedMembers().alpha)
#define BETA (getSharedMembers().beta)

#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
//START
#if RECEPTOR_AMPA == AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define NEUROTRANSMITTER      \
  (getSharedMembers().NTmax / \
   (1.0 + exp(-(*Vpre - getSharedMembers().Vp) / getSharedMembers().Kp)))

#elif RECEPTOR_AMPA == AMPAR_POINTPROCESS
// NOTE: Vp = Vthreshold
#define NEUROTRANSMITTER      \
  (((*Vpre) > getSharedMembers().Vp ? getSharedMembers().NTmax : 0))

#else
  {
  NOT SUPPORT
  }
#endif

//END
#elif SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT
#define NEUROTRANSMITTER *Glut
#endif

// take into account the effect of temperature making the change faster or
// slower
#define DT (*(getSharedMembers().deltaT))
#define Tscale (*(getSharedMembers().deltaT) * (getSharedMembers().Tadj))

void AMPAReceptor::initializeAMPA(RNG& rng)
{
#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
  assert(Vpre);
#endif
  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  r = ALPHANEUROTRANSMITTER / (BETA + ALPHANEUROTRANSMITTER);

  // std::cout << "Weight: " << (*w) << std::endl;
  if (w == NULL)
  {  // no learning rule is used
    g = gbar * r;
  }
  else
  {  // a learning rule is modeled as increasing the conductance of AMPAR
    g = (*w) * gbar * r;
  }
}

void AMPAReceptor::updateAMPA(RNG& rng)
{
  // Rempe-Chopp 2006
  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  dyn_var_t A = Tscale*(BETA + ALPHANEUROTRANSMITTER)/2.0;
  r =  (Tscale*ALPHANEUROTRANSMITTER + r*(1.0 - A))/(1.0 + A);
  
  //r = fraction of channel-opening
  if (r < 0.0)
  {
    r = 0.0;
  }
  else if (r > 1.0)
  {
    r = 1.0;
  }

  if (w == NULL)
  {
    g = gbar * r;
  }
  else
  {
    g = (*w) * gbar * r;
  }
  I = g * ((*Vpost)[indexPost] - getSharedMembers().E);
  //I = NEUROTRANSMITTER;
}

void AMPAReceptor::setPostIndex(const String& CG_direction,
                                const String& CG_component,
                                NodeDescriptor* CG_node, Edge* CG_edge,
                                VariableDescriptor* CG_variable,
                                Constant* CG_constant,
                                CG_AMPAReceptorInAttrPSet* CG_inAttrPset,
                                CG_AMPAReceptorOutAttrPSet* CG_outAttrPset)
{
  indexPost = CG_inAttrPset->idx;
  if (indexPrePost.size() % 2)
  {//it means that PreSynapticPoint is being used
#ifdef KEEP_PAIR_PRE_POST
    indexPrePost.push_back(&indexPost);
#endif
  }
  if (indexPrePost.size() == 0)
  {
    //it means the SynapticCleft is on different rank
    branchDataPrePost.increaseSizeTo(2);
    branchDataPrePost[0] = (getSharedMembers().branchDataPost); // array of 2n elements; in pair (preBD, postBD)
    branchDataPrePost[1] = (getSharedMembers().branchDataPost); // array of 2n elements; in pair (preBD, postBD)
    indexPrePost.push_back(&indexPost);
    indexPrePost.push_back(&indexPost);
  }
}

AMPAReceptor::~AMPAReceptor() {}
void AMPAReceptor::setPrePostIndex(const String& CG_direction,
                                const String& CG_component,
                                NodeDescriptor* CG_node, Edge* CG_edge,
                                VariableDescriptor* CG_variable,
                                Constant* CG_constant,
                                CG_AMPAReceptorInAttrPSet* CG_inAttrPset,
                                CG_AMPAReceptorOutAttrPSet* CG_outAttrPset)
{
  NodeProxyBase* node = dynamic_cast<NodeProxyBase*>(CG_node->getNode());
  if (node == 0)
  {//not a proxy
    indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[0]);
    indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[1]);
  }
  else{
    //TUAN TODO: how to handle this scenario
    // when the SynapticCleft is not on the same rank
    // current solution: append the post-side for the pre-side as done in 
    //  setPostIndex() 
  }
}
