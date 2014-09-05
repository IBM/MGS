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
#include "BackwardSolvePoint3.h"
#include "CG_BackwardSolvePoint3.h"
#include "rndm.h"

void BackwardSolvePoint3::produceInitialState(RNG& rng) 
{
}

void BackwardSolvePoint3::produceArea(RNG& rng) 
{
}

void BackwardSolvePoint3::produceBackwardSolution(RNG& rng) 
{
}

void BackwardSolvePoint3::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint3InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint3OutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().solutionConnect);
  assert(getSharedMembers().solutionConnect->size()>0);    
  solution = &((*(getSharedMembers().solutionConnect))[0]);
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()>0);    
  dimension = (*(getSharedMembers().dimensionsConnect))[0];  
}

BackwardSolvePoint3::~BackwardSolvePoint3() 
{
}

