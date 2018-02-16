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

void MegaSynapticCleft::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MegaSynapticCleftInAttrPSet* CG_inAttrPset, CG_MegaSynapticCleftOutAttrPSet* CG_outAttrPset) 
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

