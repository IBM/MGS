// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "IP3ConcentrationJunctionPoint.h"
#include "CG_IP3ConcentrationJunctionPoint.h"
#include "rndm.h"

void IP3ConcentrationJunctionPoint::produceInitialState(RNG& rng) 
{
}

void IP3ConcentrationJunctionPoint::produceIP3Concentration(RNG& rng) 
{
}

void IP3ConcentrationJunctionPoint::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationJunctionPointOutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().IP3ConcentrationConnect);
  assert(getSharedMembers().IP3ConcentrationConnect->size()==1);
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()==1);    
  IP3Concentration = &((*(getSharedMembers().IP3ConcentrationConnect))[0]);
  dimension = (*(getSharedMembers().dimensionsConnect))[0];
}

IP3ConcentrationJunctionPoint::~IP3ConcentrationJunctionPoint() 
{
}

