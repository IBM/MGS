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
#include "LeakyIAFUnit.h"
#include "CG_LeakyIAFUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()

void LeakyIAFUnit::initialize(RNG& rng) 
{
  spike = 0; // start with no spike
}

void LeakyIAFUnit::update(RNG& rng) 
{
  // Add up driver ...
  double tempInput = 0.0;
  tempInput += SHD.driver;
  // ... and recurrent spike input ...
  ShallowArray<SpikeInput>::iterator iterInputs, endInputs=inputs.end();
  for (iterInputs=inputs.begin(); iterInputs!=endInputs; ++iterInputs)
    tempInput += (*(iterInputs->spike) ? 1.0 : 0.0) * iterInputs->weight;

  // ... 'and update membrane potential with it.
  V += ((-V + tempInput) / SHD.tau) * SHD.deltaT;
}

void LeakyIAFUnit::threshold(RNG& rng) 
{
  // Threshold and update spike appropriately
  if (V >= SHD.threshold)
    {
      V = 0.0;
      spike = 1; // set to a spike
    }
  else
    spike = 0;   // reset to no spike  
}

void LeakyIAFUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LeakyIAFUnitInAttrPSet* CG_inAttrPset, CG_LeakyIAFUnitOutAttrPSet* CG_outAttrPset) 
{
}

LeakyIAFUnit::~LeakyIAFUnit() 
{
}

