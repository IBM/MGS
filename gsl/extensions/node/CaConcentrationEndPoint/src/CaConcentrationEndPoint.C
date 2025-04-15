// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "CaConcentrationEndPoint.h"
#include "CG_CaConcentrationEndPoint.h"
#include "rndm.h"

void CaConcentrationEndPoint::produceInitialState(RNG& rng) 
{
}

void CaConcentrationEndPoint::produceSolvedCaConcentration(RNG& rng) 
{
}

void CaConcentrationEndPoint::produceFinishedCaConcentration(RNG& rng) 
{
}

void CaConcentrationEndPoint::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_CaConcentrationEndPointOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->identifier=="distalEnd") {
    assert(getSharedMembers().CaConcentrationConnect);
    assert(getSharedMembers().CaConcentrationConnect->size()>0);    
    CaConcentration = &((*(getSharedMembers().CaConcentrationConnect))[0]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[0];    
  }
  else if (CG_inAttrPset->identifier=="proximalEnd") {
    assert(getSharedMembers().CaConcentrationConnect);
    assert(getSharedMembers().CaConcentrationConnect->size()>0);    
    CaConcentration = &((*(getSharedMembers().CaConcentrationConnect))[getSharedMembers().CaConcentrationConnect->size()-1]);
    assert(getSharedMembers().dimensionsConnect);
    assert(getSharedMembers().dimensionsConnect->size()>0);    
    dimension = (*(getSharedMembers().dimensionsConnect))[getSharedMembers().dimensionsConnect->size()-1];
  }
  else {
    std::cerr<<"Warning! Unrecognized input identifier on CaConcentrationEndPoint!"<<std::endl;
  }
}

CaConcentrationEndPoint::~CaConcentrationEndPoint() 
{
}

