// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "CaExtrusion.h"
#include "CG_CaExtrusion.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NTSMacros.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

void CaExtrusion::initialize(RNG& rng)
{
  assert(branchData);
  unsigned size = branchData->size;
  assert(Ca_IC);
  assert(Ca_IC->size() == size);
  // allocate
  if (J_Ca.size() != size) J_Ca.increaseSizeTo(size);
  // initialize
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    J_Ca[i] = 1.0 / ((getSharedMembers().tau_pump)) * (
                      getSharedMembers().Ca_equil - cai);  // [uM/ms]
  }
}

void CaExtrusion::update(RNG& rng) {
	for (unsigned i = 0; i < branchData->size; ++i)
	{
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    J_Ca[i] = 1.0 / ((getSharedMembers().tau_pump)) * (
                      getSharedMembers().Ca_equil - cai);  // [uM/msec]
	}
}


void CaExtrusion::setPointers(const CustomString& CG_direction,
                              const CustomString& CG_component,
                              NodeDescriptor* CG_node, Edge* CG_edge,
                              VariableDescriptor* CG_variable,
                              Constant* CG_constant,
                              CG_CaExtrusionInAttrPSet* CG_inAttrPset,
                              CG_CaExtrusionOutAttrPSet* CG_outAttrPset)
{
}

CaExtrusion::~CaExtrusion() {}
