// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

