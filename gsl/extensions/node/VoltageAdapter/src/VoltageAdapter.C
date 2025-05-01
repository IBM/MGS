// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "VoltageAdapter.h"
#include "CG_VoltageAdapter.h"
#include "rndm.h"

void VoltageAdapter::produceInitialVoltage(RNG& rng) 
{
}

void VoltageAdapter::produceVoltage(RNG& rng) 
{
}


void VoltageAdapter::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageAdapterInAttrPSet* CG_inAttrPset, CG_VoltageAdapterOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->idx>=0) {
    index=CG_inAttrPset->idx;
  }
  else if (CG_inAttrPset->idx==-1) {
    index=int(float(branchData->size)*CG_inAttrPset->branchProp);
  }
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
  dimension = ((*(getSharedMembers().dimensionsConnect))[index]);
}

VoltageAdapter::~VoltageAdapter() 
{
}

