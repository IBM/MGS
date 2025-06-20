// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "MegaSynapticCleft.h"
#include "CG_MegaSynapticCleft.h"
#include "rndm.h"

void MegaSynapticCleft::produceInitialVoltage(RNG& rng) 
{
}

void MegaSynapticCleft::produceVoltage(RNG& rng) 
{
}

void MegaSynapticCleft::computeState(RNG& rng) 
{
   //aggregate all ShallowArray<dyn_var_t*> Vm
   //to produce LFP
   LFP = 0.0; 
   //std::for_each(Vm.begin(), Vm.end(), [&])
   for (auto& n : Vm)
      LFP +=  *n;
}

void MegaSynapticCleft::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MegaSynapticCleftInAttrPSet* CG_inAttrPset, CG_MegaSynapticCleftOutAttrPSet* CG_outAttrPset) 
{
  int index;
  if (CG_inAttrPset->idx>=0) {
    index=CG_inAttrPset->idx;
  }
  else if (CG_inAttrPset->idx==-1) {
    //index=int(float(branchData->size)*CG_inAttrPset->branchProp);
    index=int(float(getSharedMembers().branchData->size)*CG_inAttrPset->branchProp);
  }
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());
  //Vi = &((*(getSharedMembers().voltageConnect))[index]);
  Vm.push_back(&((*(getSharedMembers().voltageConnect))[index]));
}

MegaSynapticCleft::~MegaSynapticCleft() 
{
}

