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
#include "IP3ConcentrationJunctionPoint.h"
#include "CG_IP3ConcentrationJunctionPoint.h"
#include "rndm.h"

void IP3ConcentrationJunctionPoint::produceInitialState(RNG& rng) 
{
}

void IP3ConcentrationJunctionPoint::produceIP3Concentration(RNG& rng) 
{
}

void IP3ConcentrationJunctionPoint::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationJunctionPointOutAttrPSet* CG_outAttrPset) 
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

