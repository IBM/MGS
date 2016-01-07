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
#include "Connexon_Vm.h"
#include "CG_Connexon_Vm.h"
#include "rndm.h"

void Connexon_Vm::produceInitialState(RNG& rng) 
{
}

void Connexon_Vm::produceState(RNG& rng) 
{
}

void Connexon_Vm::computeState(RNG& rng) 
{
  I=g*(*Vj-*Vi);
}

void Connexon_Vm::setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_Connexon_VmInAttrPSet* CG_inAttrPset, CG_Connexon_VmOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
}

Connexon_Vm::~Connexon_Vm() 
{
}

