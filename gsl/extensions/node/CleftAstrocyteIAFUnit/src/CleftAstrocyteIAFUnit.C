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
#include "CleftAstrocyteIAFUnit.h"
#include "CG_CleftAstrocyteIAFUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()

void CleftAstrocyteIAFUnit::initialize(RNG& rng)
{
  // Default starting values
  glutamate = 0.0;
<<<<<<< HEAD
=======
  ECB = 0.0;
>>>>>>> origin/team-A
}

void CleftAstrocyteIAFUnit::update(RNG& rng)
{
  // Increase glutamate concentration in the cleft due to pre-synaptic release
  if (glutamateInput.size() > 0)
    glutamate += *(glutamateInput[0].glutamate) * glutamateInput[0].weight; // only consider first one, weight is structural plasticity
  // Astrocyte reuptake of glutamate with GLT-1
  glutamate += (-glutamate / SHD.glutamateDecayTau) * SHD.deltaT;
<<<<<<< HEAD
=======

  // ECB diffuses really quickly, so this level is equal to that produced by the spine
  if (ECBInput.size() > 0)
    ECB = *(ECBInput[0].ECB) * ECBInput[0].weight; // only consider first one, weight is structural plasticity
>>>>>>> origin/team-A
}

void CleftAstrocyteIAFUnit::setGlutamateIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset)
{
  glutamateInput[glutamateInput.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab
  glutamateInput[glutamateInput.size()-1].col = CG_node->getGlobalIndex()+1;
}

<<<<<<< HEAD
=======
void CleftAstrocyteIAFUnit::setECBIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset)
{
  ECBInput[ECBInput.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab
  ECBInput[ECBInput.size()-1].col = CG_node->getGlobalIndex()+1;
}

>>>>>>> origin/team-A
CleftAstrocyteIAFUnit::~CleftAstrocyteIAFUnit()
{
}

