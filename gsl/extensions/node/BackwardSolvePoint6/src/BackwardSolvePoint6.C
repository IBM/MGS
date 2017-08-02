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
#include "BackwardSolvePoint6.h"
#include "CG_BackwardSolvePoint6.h"
#include "rndm.h"

void BackwardSolvePoint6::produceInitialState(RNG& rng) 
{
}

void BackwardSolvePoint6::produceArea(RNG& rng) 
{
}

void BackwardSolvePoint6::produceBackwardSolution(RNG& rng) 
{
}

void BackwardSolvePoint6::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint6InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint6OutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().solutionConnect);
  assert(getSharedMembers().solutionConnect->size()>0);    
  solution = &((*(getSharedMembers().solutionConnect))[0]);
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()>0);    
  dimension = (*(getSharedMembers().dimensionsConnect))[0];  
}

BackwardSolvePoint6::~BackwardSolvePoint6() 
{
}

