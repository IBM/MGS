// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

