// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FSIIAFUnit_H
#define FSIIAFUnit_H

#include "Lens.h"
#include "CG_FSIIAFUnit.h"
#include "rndm.h"
#include <fstream>

class FSIIAFUnit : public CG_FSIIAFUnit
{
   public:
      void initialize(RNG& rng);
      void updateInput(RNG& rng);
      void updateV(RNG& rng);
      void threshold(RNG& rng);
      void outputPSPs(std::ofstream& fs);
      void outputWeights(std::ofstream& fs);
      void outputGJs(std::ofstream& fs);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual bool bidirectional1(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual bool bidirectional2(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~FSIIAFUnit();
};

#endif
