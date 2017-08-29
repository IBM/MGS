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
#include "CleftAstrocyteIAFUnit.h"
#include "CG_CleftAstrocyteIAFUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()

void CleftAstrocyteIAFUnit::initialize(RNG& rng)
{
  // Check if more than one input
  if (neurotransmitterInput.size() != 1)
    assert("CleftAstrocyteIAFUnit: neurotransmitter inputs should be one.");
  if (ECBInput.size() != 1)
    assert("CleftAstrocyteIAFUnit: ECB inputs should be one.");
  // Default starting values
  neurotransmitter = 0.0;
  ECB = 0.0;
}

void CleftAstrocyteIAFUnit::update(RNG& rng)
{
  // Increase neurotransmitter concentration in the cleft due to pre-synaptic release
  if (neurotransmitterInput.size() > 0)
    neurotransmitter += *(neurotransmitterInput[0].neurotransmitter) * neurotransmitterInput[0].weight; // only consider first one, weight is structural plasticity
  // Astrocyte reuptake of neurotransmitter with GLT-1
  neurotransmitter += (-neurotransmitter / SHD.neurotransmitterDecayTau) * SHD.deltaT;

  // ECB diffuses really quickly, so this level is equal to that produced by the spine
  if (ECBInput.size() > 0)
    ECB = *(ECBInput[0].ECB) * ECBInput[0].weight; // only consider first one, weight is structural plasticity
}

void CleftAstrocyteIAFUnit::setNeurotransmitterIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset)
{
  neurotransmitterInput[neurotransmitterInput.size()-1].row =  getIndex()+1; // +1 is for Matlab
  neurotransmitterInput[neurotransmitterInput.size()-1].col = CG_node->getIndex()+1;
}

void CleftAstrocyteIAFUnit::setECBIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset)
{
  ECBInput[ECBInput.size()-1].row =  getIndex()+1; // +1 is for Matlab
  ECBInput[ECBInput.size()-1].col = CG_node->getIndex()+1;
}

CleftAstrocyteIAFUnit::~CleftAstrocyteIAFUnit()
{
}

