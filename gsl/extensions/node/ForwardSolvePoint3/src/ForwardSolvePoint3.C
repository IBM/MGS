// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "ForwardSolvePoint3.h"
#include "CG_ForwardSolvePoint3.h"
#include "rndm.h"

void ForwardSolvePoint3::produceInitialState(RNG& rng) 
{
}

void ForwardSolvePoint3::produceInitialCoefficients(RNG& rng) 
{
}

void ForwardSolvePoint3::produceForwardSolution(RNG& rng) 
{
}

void ForwardSolvePoint3::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint3InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint3OutAttrPSet* CG_outAttrPset) 
{
  assert(getSharedMembers().dimensionsConnect);
  assert(getSharedMembers().dimensionsConnect->size()>0);    
  dimension = (*(getSharedMembers().dimensionsConnect))[getSharedMembers().dimensionsConnect->size()-1];

  assert(getSharedMembers().AiiConnect);
  assert(getSharedMembers().AiiConnect->size()>0);    
  Aii = &((*(getSharedMembers().AiiConnect))[getSharedMembers().AiiConnect->size()-1]);

  assert(getSharedMembers().AipConnect);
  assert(getSharedMembers().AipConnect->size()>0);    
  Aip = &((*(getSharedMembers().AipConnect))[getSharedMembers().AipConnect->size()-1]);
  
  assert(getSharedMembers().RHSConnect);
  assert(getSharedMembers().RHSConnect->size()>0);    
  RHS = &((*(getSharedMembers().RHSConnect))[getSharedMembers().RHSConnect->size()-1]);
}

ForwardSolvePoint3::~ForwardSolvePoint3() 
{
}

