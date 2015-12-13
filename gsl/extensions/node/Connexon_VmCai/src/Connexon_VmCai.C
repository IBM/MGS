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
#include "Connexon_VmCai.h"
#include "CG_Connexon_VmCai.h"
#include "rndm.h"

void Connexon_VmCai::produceInitialState(RNG& rng) 
{
}

void Connexon_VmCai::produceState(RNG& rng) 
{
}

void Connexon_VmCai::computeState(RNG& rng) 
{
  float V=*Vj-*Vi;
  I=g*V;
  float E_Ca=0.08686 * *(getSharedMembers().T) * log(*Caj / *Cai);
  I_Ca=gMYO*(V+E_Ca);
}

void Connexon_VmCai::setCaPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_Connexon_VmCaiInAttrPSet* CG_inAttrPset, CG_Connexon_VmCaiOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().CaConcentrationConnect);
  assert(index>=0 && index<getSharedMembers().CaConcentrationConnect->size());    
  Cai = &((*(getSharedMembers().CaConcentrationConnect))[index]);
}

void Connexon_VmCai::setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_Connexon_VmCaiInAttrPSet* CG_inAttrPset, CG_Connexon_VmCaiOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
}

Connexon_VmCai::~Connexon_VmCai() 
{
}

