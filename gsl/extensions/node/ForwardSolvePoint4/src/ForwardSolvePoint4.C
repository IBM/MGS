// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "ForwardSolvePoint4.h"
#include "CG_ForwardSolvePoint4.h"
#include "rndm.h"

void ForwardSolvePoint4::produceInitialState(RNG& rng) 
{
}

void ForwardSolvePoint4::produceInitialCoefficients(RNG& rng) 
{
}

void ForwardSolvePoint4::produceForwardSolution(RNG& rng) 
{
}

void ForwardSolvePoint4::setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint4InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint4OutAttrPSet* CG_outAttrPset) 
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

ForwardSolvePoint4::~ForwardSolvePoint4() 
{
}

