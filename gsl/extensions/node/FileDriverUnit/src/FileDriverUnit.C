// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "FileDriverUnit.h"
#include "CG_FileDriverUnit.h"
#include "rndm.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()

void FileDriverUnit::initialize(RNG& rng) 
{  
  // Set channels to node index, but adjust in case there are not enough channels
  if (SHD.op_random)
    channel = irandom(0, SHD.n_channels - 1, rng);
  else
    channel = getGlobalIndex() % SHD.n_channels;
}

void FileDriverUnit::updateOutput(RNG& rng) 
{
  if (ITER % SHD.period == 0)
    output = SHD.input[channel] * SHD.scale;
}

void FileDriverUnit::setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FileDriverUnitInAttrPSet* CG_inAttrPset, CG_FileDriverUnitOutAttrPSet* CG_outAttrPset) 
{
}

FileDriverUnit::~FileDriverUnit() 
{
}

