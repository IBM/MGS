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
#include "SpineAttachment_VmCai.h"
#include "CG_SpineAttachment_VmCai.h"
#include "rndm.h"

void SpineAttachment_VmCai::produceInitialState(RNG& rng) 
{
	//NOTE: g (nS) which is infered from R (GigaOhm)
	//   R = rho. l / A
	//   rho (GigaOhm.cm) = specific resisitivity
	//   l = length of diffusion
	//   A = cross-sectional area
	//   r1 = radius of spine neck
	//   r2 = radius of dendritic shaft
	//  with complex geometry, calculating the resistance be much more complicated
	//  https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity#Resistance_versus_resistivity_in_complicated_geometries
	//  SOLUTION: l = distance from 2 center points in neck + compartment
	//     l = 1/2 nec-length + r2 
	//            A = pi * ((r1+r2)/2)^2
	//            rho = Ra ~ 100 GOhm.um
	//  g = 1/R = A / (rho * l) 
	g = abs(Ai - Aj) / (Raxial * (leni + lenj)/2.0); // [nS]
	gCYTO = abs(Ai - Aj) / (RCacytoaxial * (leni + lenj)/2.0); // [nS]
}

void SpineAttachment_VmCai::produceState(RNG& rng) 
{
}

void SpineAttachment_VmCai::computeState(RNG& rng) 
{
  float V=*Vj-*Vi;
  I=g*V;
  float E_Ca=0.08686 * *(getSharedMembers().T) * log(*Caj / *Cai);
  I_Ca=gCYTO*(V+E_Ca);
}

void SpineAttachment_VmCai::setCaPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().CaConcentrationConnect);
  assert(index>=0 && index<getSharedMembers().CaConcentrationConnect->size());    
  Cai = &((*(getSharedMembers().CaConcentrationConnect))[index]);
}

void SpineAttachment_VmCai::setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
}

void SpineAttachment_VmCai::set_A_and_len(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset) 
{
  Ai=CG_inAttrPset->A;
  leni = CG_inAttrPset->len;
}

SpineAttachment_VmCai::~SpineAttachment_VmCai() 
{
}

