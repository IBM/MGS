// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "ToneUnit.h"
#include "CG_ToneUnit.h"
#include "rndm.h"

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()

void ToneUnit::initialize(RNG& rng) 
{
}

void ToneUnit::update(RNG& rng) 
{
  if (ITER>=SHD.start_time && ITER<SHD.stop_time) {
    if ((ITER % SHD.interval)==0) {
      tone = SHD.amplitude;
    } else if (ITER % SHD.interval > SHD.duration) {
      tone = 0;
    }
  } else {
    tone = 0;
  }
}

void ToneUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ToneUnitInAttrPSet* CG_inAttrPset, CG_ToneUnitOutAttrPSet* CG_outAttrPset) 
{
}

ToneUnit::~ToneUnit() 
{
}

