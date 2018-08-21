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
#include "GABAAReceptor.h"
#include "CG_GABAAReceptor.h"
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
#define NEUROTRANSMITTER      \
  (getSharedMembers().NTmax / \
   (1.0 + exp(-(*Vpre - getSharedMembers().Vp) / getSharedMembers().Kp)))

#elif SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT
#define NEUROTRANSMITTER *GABA
#endif

#define DT (*(getSharedMembers().deltaT))
#define Tscale (*(getSharedMembers().deltaT) * (getSharedMembers().Tadj))

void GABAAReceptor::initializeGABAA(RNG& rng)
{
#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
  assert(Vpre);
#endif
  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  r = ALPHANEUROTRANSMITTER / (BETA + ALPHANEUROTRANSMITTER);
  g = gbar * r;
}

void GABAAReceptor::updateGABAA(RNG& rng)
{
  // Rempe-Chopp 2006
  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  dyn_var_t A = Tscale*(BETA + ALPHANEUROTRANSMITTER)/2.0;
  r =  (Tscale*ALPHANEUROTRANSMITTER + r*(1.0 - A))/(1.0 + A);
  //dyn_var_t tmp = 1.0 + (ALPHANEUROTRANSMITTER + BETA) / 2.0 * Tscale;
  //r = (r * tmp + ALPHANEUROTRANSMITTER * Tscale) / tmp;
  g = gbar * r;
  I = g * ((*Vpost)[indexPost] - getSharedMembers().E);
}

void GABAAReceptor::setPostIndex(const String& CG_direction,
                                 const String& CG_component,
                                 NodeDescriptor* CG_node, Edge* CG_edge,
                                 VariableDescriptor* CG_variable,
                                 Constant* CG_constant,
                                 CG_GABAAReceptorInAttrPSet* CG_inAttrPset,
                                 CG_GABAAReceptorOutAttrPSet* CG_outAttrPset)
{
  if (CG_inAttrPset->idx>=0) {
    indexPost = CG_inAttrPset->idx;
  }
  else if (CG_inAttrPset->idx==-1) {
    indexPost=int(float(branchDataPrePost[branchDataPrePost.size()-1]->size)*CG_inAttrPset->branchProp);
  }
  //indexPrePost.push_back(&indexPost);
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

void GABAAReceptor::setPrePostIndex(const String& CG_direction,
    const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge,
    VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_GABAAReceptorInAttrPSet* CG_inAttrPset,
    CG_GABAAReceptorOutAttrPSet* CG_outAttrPset)
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

GABAAReceptor::~GABAAReceptor() {}
