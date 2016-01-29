// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2015-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "SpineAttachment_VmCaiCaER.h"
#include "CG_SpineAttachment_VmCaiCaER.h"
#include "rndm.h"

void SpineAttachment_VmCaiCaER::produceInitialState(RNG& rng) 
{
}

void SpineAttachment_VmCaiCaER::produceState(RNG& rng) 
{
}

void SpineAttachment_VmCaiCaER::computeState(RNG& rng) 
{
  float V=*Vj-*Vi;
  I=g*V;
  float E_Ca=0.08686 * *(getSharedMembers().T) * log(*Caj / *Cai);
  I_Ca=gCYTO*(V+E_Ca);
  float E_CaER=0.08686 * *(getSharedMembers().T) * log(*CaERj / *CaERi);
  I_CaER=gER*(E_CaER);
}

void SpineAttachment_VmCaiCaER::setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset) 
{
  int index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
}

void SpineAttachment_VmCaiCaER::setCaPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().CaConcentrationConnect);
  assert(index>=0 && index<getSharedMembers().CaConcentrationConnect->size());    
  Cai = &((*(getSharedMembers().CaConcentrationConnect))[index]);
}

void SpineAttachment_VmCaiCaER::setCaERPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().CaERConcentrationConnect);
  assert(index>=0 && index<getSharedMembers().CaERConcentrationConnect->size());    
  CaERi = &((*(getSharedMembers().CaERConcentrationConnect))[index]);
}

SpineAttachment_VmCaiCaER::~SpineAttachment_VmCaiCaER() 
{
}

