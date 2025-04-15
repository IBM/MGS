// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RabinovichWinnerlessUnit_H
#define RabinovichWinnerlessUnit_H

#include "Lens.h"
#include "CG_RabinovichWinnerlessUnit.h"
#include "rndm.h"
#include <fstream>

class RabinovichWinnerlessUnit : public CG_RabinovichWinnerlessUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void copy(RNG& rng);
      void outputWeights(std::ofstream& fsLN, std::ofstream& fsDR, std::ofstream& fsNS);
      virtual void checkForCorticalSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual void setLateralIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual void setModulatoryIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual void assymetric(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual ~RabinovichWinnerlessUnit();
};

#endif
