// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "VoltageJunctionPoint.h"
#include "CG_VoltageJunctionPoint.h"
#include "rndm.h"

void VoltageJunctionPoint::produceInitialState(RNG& rng) 
{
}

void VoltageJunctionPoint::produceVoltage(RNG& rng) 
{
}

void VoltageJunctionPoint::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageJunctionPointInAttrPSet* CG_inAttrPset, CG_VoltageJunctionPointOutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().voltageConnect);
  assert(getSharedMembers().voltageConnect->size()==1);    
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()==1);    
  voltage = &((*(getSharedMembers().voltageConnect))[0]);
  dimension = (*(getSharedMembers().dimensionsConnect))[0];
}

VoltageJunctionPoint::~VoltageJunctionPoint() 
{
}

