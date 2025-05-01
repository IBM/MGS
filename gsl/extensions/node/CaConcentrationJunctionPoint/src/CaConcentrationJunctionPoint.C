// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "CaConcentrationJunctionPoint.h"
#include "CG_CaConcentrationJunctionPoint.h"
#include "rndm.h"

void CaConcentrationJunctionPoint::produceInitialState(RNG& rng) 
{
}

void CaConcentrationJunctionPoint::produceCaConcentration(RNG& rng) 
{
}

void CaConcentrationJunctionPoint::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionPointOutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().CaConcentrationConnect);
  assert(getSharedMembers().CaConcentrationConnect->size()==1);
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()==1);    
  CaConcentration = &((*(getSharedMembers().CaConcentrationConnect))[0]);
  dimension = (*(getSharedMembers().dimensionsConnect))[0];
}

CaConcentrationJunctionPoint::~CaConcentrationJunctionPoint() 
{
}

