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
#include "BackwardSolvePoint5.h"
#include "CG_BackwardSolvePoint5.h"
#include "rndm.h"

void BackwardSolvePoint5::produceInitialState(RNG& rng) 
{
}

void BackwardSolvePoint5::produceArea(RNG& rng) 
{
}

void BackwardSolvePoint5::produceBackwardSolution(RNG& rng) 
{
}

void BackwardSolvePoint5::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint5InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint5OutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().solutionConnect);
  assert(getSharedMembers().solutionConnect->size()>0);    
  solution = &((*(getSharedMembers().solutionConnect))[0]);
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()>0);    
  dimension = (*(getSharedMembers().dimensionsConnect))[0];  
}

BackwardSolvePoint5::~BackwardSolvePoint5() 
{
}

