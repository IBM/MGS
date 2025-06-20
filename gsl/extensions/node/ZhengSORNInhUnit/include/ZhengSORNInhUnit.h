// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ZhengSORNInhUnit_H
#define ZhengSORNInhUnit_H

#include "Mgs.h"
#include "CG_ZhengSORNInhUnit.h"
#include "rndm.h"
#include <fstream>

class ZhengSORNInhUnit : public CG_ZhengSORNInhUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void fire(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNInhUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNInhUnitOutAttrPSet* CG_outAttrPset);
      void inputWeights(std::ifstream& fsE2I, int col, float weight);
      void outputWeights(std::ofstream& fsE2I);
      void inputTI(float val);
      virtual ~ZhengSORNInhUnit();
};

#endif
