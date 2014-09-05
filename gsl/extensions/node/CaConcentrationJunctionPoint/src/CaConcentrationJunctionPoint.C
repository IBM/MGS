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
#include "CaConcentrationJunctionPoint.h"
#include "CG_CaConcentrationJunctionPoint.h"
#include "rndm.h"

void CaConcentrationJunctionPoint::produceInitialState(RNG& rng) 
{
}

void CaConcentrationJunctionPoint::produceCaConcentration(RNG& rng) 
{
}

void CaConcentrationJunctionPoint::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionPointOutAttrPSet* CG_outAttrPset) 
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

