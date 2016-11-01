// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
   (1.0 + exp(-(*V - getSharedMembers().Vp) / getSharedMembers().Kp)))
#elif SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT
#define NEUROTRANSMITTER *GABA
#endif

#define DT (*(getSharedMembers().deltaT))
#define Tscale (*(getSharedMembers().deltaT) * (getSharedMembers().Tadj))

void GABAAReceptor::initializeGABAA(RNG& rng)
{
  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  r = ALPHANEUROTRANSMITTER / (BETA + ALPHANEUROTRANSMITTER);
  g = gbar * r;
}

void GABAAReceptor::updateGABAA(RNG& rng)
{
  dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  dyn_var_t tmp = 1.0 + (ALPHANEUROTRANSMITTER + BETA) / 2.0 * Tscale;
  r = (r * tmp + ALPHANEUROTRANSMITTER * Tscale) / tmp;
  // dyn_var_t A = Tscale*(BETA + ALPHANEUROTRANSMITTER)/2.0;
  // r =  (Tscale*ALPHANEUROTRANSMITTER + r*(1.0 - A))/(1.0 + A);
  g = gbar * r;
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
  indexPrePost.push_back(&indexPost);
}

GABAAReceptor::~GABAAReceptor() {}
