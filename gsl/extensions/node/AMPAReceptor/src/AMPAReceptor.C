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
#include "AMPAReceptor.h"
#include "CG_AMPAReceptor.h"
#include "rndm.h"
#include <iostream>

#define ALPHA (getSharedMembers().alpha)
#define BETA (getSharedMembers().beta)
#define NEUROTRANSMITTER (getSharedMembers().Tmax/(1.0 + exp(-(*V - getSharedMembers().Vp)/getSharedMembers().Kp)))
#define DT (*(getSharedMembers().deltaT))

void AMPAReceptor::initializeAMPA(RNG& rng) {
  assert(V);
  float ALPHANEUROTRANSMITTER = ALPHA*NEUROTRANSMITTER;
  r = ALPHANEUROTRANSMITTER/(BETA + ALPHANEUROTRANSMITTER);

  //std::cout << "Weight: " << (*wv << std::endl;
  if(w==NULL){
    g = gbar*r;}
  else{
    g = (*w)*gbar*r;
  }
}

void AMPAReceptor::updateAMPA(RNG& rng) {
  float ALPHANEUROTRANSMITTER = ALPHA*NEUROTRANSMITTER;
  float A = DT*(BETA + ALPHANEUROTRANSMITTER)/2.0;
  r =  (DT*ALPHANEUROTRANSMITTER + r*(1.0 - A))/(1.0 + A);

  if(w==NULL){
    g = gbar*r;}
  else{
    g = (*w)*gbar*r;
  }
}

void AMPAReceptor::setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AMPAReceptorInAttrPSet* CG_inAttrPset, CG_AMPAReceptorOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->idx>=0) {
    indexPost = CG_inAttrPset->idx;
  }
  else if (CG_inAttrPset->idx==-1) {
    indexPost=int(float(branchDataPrePost[branchDataPrePost.size()-1]->size)*CG_inAttrPset->branchProp);
  }
  indexPrePost.push_back(&indexPost);
}

AMPAReceptor::~AMPAReceptor() 
{
}
