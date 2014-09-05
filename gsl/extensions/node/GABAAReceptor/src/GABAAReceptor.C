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

#define ALPHA (getSharedMembers().alpha)
#define BETA (getSharedMembers().beta)
#define NEUROTRANSMITTER (getSharedMembers().Tmax/(1.0 + exp(-(*V - getSharedMembers().Vp)/getSharedMembers().Kp)))
#define DT (*(getSharedMembers().deltaT))

void GABAAReceptor::initializeGABAA(RNG& rng) {
  float ALPHANEUROTRANSMITTER = ALPHA*NEUROTRANSMITTER;
  r = ALPHANEUROTRANSMITTER/(BETA + ALPHANEUROTRANSMITTER);
  g = gbar*r;
}

void GABAAReceptor::updateGABAA(RNG& rng) {
  float ALPHANEUROTRANSMITTER = ALPHA*NEUROTRANSMITTER;
  float A = DT*(BETA + ALPHANEUROTRANSMITTER)/2.0;
  r =  (DT*ALPHANEUROTRANSMITTER + r*(1.0 - A))/(1.0 + A);
  g = gbar*r;
}

void GABAAReceptor::setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GABAAReceptorInAttrPSet* CG_inAttrPset, CG_GABAAReceptorOutAttrPSet* CG_outAttrPset) 
{
  indexPost = CG_inAttrPset->idx;
  indexPrePost.push_back(&indexPost);
}

GABAAReceptor::~GABAAReceptor() 
{
}
