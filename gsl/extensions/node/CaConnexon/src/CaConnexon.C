// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "CaConnexon.h"
#include "CG_CaConnexon.h"
#include "rndm.h"

void CaConnexon::produceInitialState(RNG& rng) 
{
}

void CaConnexon::produceState(RNG& rng) 
{
}

void CaConnexon::computeState(RNG& rng) 
{
  float V=*Vj-*Vi;
  I=g*V;
  float E_Ca=0.08686 * *(getSharedMembers().T) * log(*Caj / *Cai);
  I_Ca=g*(V+E_Ca);
}

void CaConnexon::setCaPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConnexonInAttrPSet* CG_inAttrPset, CG_CaConnexonOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->idx>=0) {
    index=CG_inAttrPset->idx;
  }
  else if (CG_inAttrPset->idx==-1) {
    index=int(float(branchData->size)*CG_inAttrPset->branchProp);
  }
  assert(getSharedMembers().CaConcentrationConnect);
  assert(index>=0 && index<getSharedMembers().CaConcentrationConnect->size());    
  Cai = &((*(getSharedMembers().CaConcentrationConnect))[index]);
}

void CaConnexon::setVoltagePointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConnexonInAttrPSet* CG_inAttrPset, CG_CaConnexonOutAttrPSet* CG_outAttrPset) 
{
  int index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
}

CaConnexon::~CaConnexon() 
{
}

