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
#include "SpineAttachment_Vm.h"
#include "CG_SpineAttachment_Vm.h"
#include "rndm.h"

void SpineAttachment_Vm::produceInitialState(RNG& rng) 
{
}

void SpineAttachment_Vm::produceState(RNG& rng) 
{
}

void SpineAttachment_Vm::computeState(RNG& rng) 
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
	//  g = 1/R = A * rho / l 
  I=g*(*Vj-*Vi);// g = should be a function of spineneck size
}

void SpineAttachment_Vm::setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vi = &((*(getSharedMembers().voltageConnect))[index]);
}

SpineAttachment_Vm::~SpineAttachment_Vm() 
{
}

