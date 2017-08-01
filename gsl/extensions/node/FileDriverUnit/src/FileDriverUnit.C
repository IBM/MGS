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

void FileDriverUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FileDriverUnitInAttrPSet* CG_inAttrPset, CG_FileDriverUnitOutAttrPSet* CG_outAttrPset) 
{
}

FileDriverUnit::~FileDriverUnit() 
{
}

