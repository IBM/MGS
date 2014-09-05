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
#include "VoltageJunctionPoint.h"
#include "CG_VoltageJunctionPoint.h"
#include "rndm.h"

void VoltageJunctionPoint::produceInitialState(RNG& rng) 
{
}

void VoltageJunctionPoint::produceVoltage(RNG& rng) 
{
}

void VoltageJunctionPoint::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageJunctionPointInAttrPSet* CG_inAttrPset, CG_VoltageJunctionPointOutAttrPSet* CG_outAttrPset) 
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

