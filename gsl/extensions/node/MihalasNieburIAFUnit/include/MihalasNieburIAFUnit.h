// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MihalasNieburIAFUnit_H
#define MihalasNieburIAFUnit_H

#include "Lens.h"
#include "CG_MihalasNieburIAFUnit.h"
#include "rndm.h"
#include <fstream>

class MihalasNieburIAFUnit : public CG_MihalasNieburIAFUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void threshold(RNG& rng);
      void outputWeights(std::ofstream& fs);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburIAFUnitInAttrPSet* CG_inAttrPset, CG_MihalasNieburIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~MihalasNieburIAFUnit();
};

#endif
