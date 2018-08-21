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
#include "PreSynapticPoint.h"
#include "CG_PreSynapticPoint.h"
#include "rndm.h"

void PreSynapticPoint::produceInitialState(RNG& rng) 
{
}

void PreSynapticPoint::produceState(RNG& rng) 
{
}

void PreSynapticPoint::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_PreSynapticPointInAttrPSet* CG_inAttrPset, CG_PreSynapticPointOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->idx>=0) {
    index=CG_inAttrPset->idx;
  }
  else if (CG_inAttrPset->idx==-1) {
    index=int(float(branchData.size)*CG_inAttrPset->branchProp);
  }
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  voltage = &((*(getSharedMembers().voltageConnect))[index]);
  if (getSharedMembers().branchDataConnect) branchData=*(getSharedMembers().branchDataConnect);
}

PreSynapticPoint::~PreSynapticPoint() 
{
}

