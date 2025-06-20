// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
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

void BackwardSolvePoint6::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint6InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint6OutAttrPSet* CG_outAttrPset) 
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

