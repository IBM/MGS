// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "IP3ConcentrationEndPoint.h"
#include "CG_IP3ConcentrationEndPoint.h"
#include "rndm.h"

void IP3ConcentrationEndPoint::produceInitialState(RNG& rng) 
{
}

void IP3ConcentrationEndPoint::produceSolvedIP3Concentration(RNG& rng) 
{
}

void IP3ConcentrationEndPoint::produceFinishedIP3Concentration(RNG& rng) 
{
}

void IP3ConcentrationEndPoint::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationEndPointOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->identifier=="distalEnd") {
    assert(getSharedMembers().IP3ConcentrationConnect);
    assert(getSharedMembers().IP3ConcentrationConnect->size()>0);    
    IP3Concentration = &((*(getSharedMembers().IP3ConcentrationConnect))[0]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[0];    
  }
  else if (CG_inAttrPset->identifier=="proximalEnd") {
    assert(getSharedMembers().IP3ConcentrationConnect);
    assert(getSharedMembers().IP3ConcentrationConnect->size()>0);    
    IP3Concentration = &((*(getSharedMembers().IP3ConcentrationConnect))[getSharedMembers().IP3ConcentrationConnect->size()-1]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[getSharedMembers().dimensionsConnect->size()-1];
  }
  else {
    std::cerr<<"Warning! Unrecognized input identifier on IP3ConcentrationEndPoint!"<<std::endl;
  }
}

IP3ConcentrationEndPoint::~IP3ConcentrationEndPoint() 
{
}

