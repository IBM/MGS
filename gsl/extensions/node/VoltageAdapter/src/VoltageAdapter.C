#include "Lens.h"
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

