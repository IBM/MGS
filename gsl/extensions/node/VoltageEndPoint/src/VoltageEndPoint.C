// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "VoltageEndPoint.h"
#include "CG_VoltageEndPoint.h"
#include "rndm.h"

void VoltageEndPoint::produceInitialState(RNG& rng) 
{
}

void VoltageEndPoint::produceSolvedVoltage(RNG& rng) 
{
}

void VoltageEndPoint::produceFinishedVoltage(RNG& rng) 
{
}

void VoltageEndPoint::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageEndPointInAttrPSet* CG_inAttrPset, CG_VoltageEndPointOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->identifier=="distalEnd") {
    assert(getSharedMembers().voltageConnect);
    assert(getSharedMembers().voltageConnect->size()>0);    
    voltage = &((*(getSharedMembers().voltageConnect))[0]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[0];    
  }
  else if (CG_inAttrPset->identifier=="proximalEnd") {
    assert(getSharedMembers().voltageConnect);
    assert(getSharedMembers().voltageConnect->size()>0);    
    voltage = &((*(getSharedMembers().voltageConnect))[getSharedMembers().voltageConnect->size()-1]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[getSharedMembers().dimensionsConnect->size()-1];
  }
  else {
    std::cerr<<"Warning! Unrecognized input identifier on VoltageEndPoint!"<<std::endl;
  }
}

VoltageEndPoint::~VoltageEndPoint() 
{
}

