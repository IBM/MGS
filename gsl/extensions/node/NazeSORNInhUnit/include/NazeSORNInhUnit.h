// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NazeSORNInhUnit_H
#define NazeSORNInhUnit_H

#include "Mgs.h"
#include "CG_NazeSORNInhUnit.h"
#include "rndm.h"
#include <fstream>

class NazeSORNInhUnit : public CG_NazeSORNInhUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void fire(RNG& rng);
      virtual void setExcIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNInhUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNInhUnitOutAttrPSet* CG_outAttrPset);
      virtual void setInhIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNInhUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNInhUnitOutAttrPSet* CG_outAttrPset);
      void inputWeights(std::ifstream& fsE2I, int col, float weight);
      void outputWeights(std::ofstream& fsE2I);
      void inputTI(float val);
      virtual ~NazeSORNInhUnit();
};

#endif
