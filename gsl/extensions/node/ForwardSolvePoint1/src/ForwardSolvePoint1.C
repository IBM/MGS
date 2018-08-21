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
#include "ForwardSolvePoint1.h"
#include "CG_ForwardSolvePoint1.h"
#include "rndm.h"

void ForwardSolvePoint1::produceInitialState(RNG& rng) 
{
}

void ForwardSolvePoint1::produceInitialCoefficients(RNG& rng) 
{
}

void ForwardSolvePoint1::produceForwardSolution(RNG& rng) 
{
}

void ForwardSolvePoint1::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint1InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint1OutAttrPSet* CG_outAttrPset) 
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

ForwardSolvePoint1::~ForwardSolvePoint1() 
{
}

