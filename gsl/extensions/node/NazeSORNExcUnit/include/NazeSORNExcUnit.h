// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NazeSORNExcUnit_H
#define NazeSORNExcUnit_H

#include "Lens.h"
#include "CG_NazeSORNExcUnit.h"
#include "rndm.h"
#include <fstream>

class NazeSORNExcUnit : public CG_NazeSORNExcUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void fire(RNG& rng);
      virtual void checkForSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void checkForInhSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void setE2EIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void setI2EIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual bool checkInitWeights(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNExcUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void outputWeights(std::ofstream& fsE2E, std::ofstream& fsI2E);
      virtual void outputDelays(std::ofstream& fsE2Ed);
      virtual void inputWeights(int col, float weight);
      virtual void inputI2EWeights(int col, float weight);
      void inputTE(float val);
      virtual void getInitParams(std::ofstream& fs_etaIP, std::ofstream& fs_HIP);
      virtual ~NazeSORNExcUnit();
};

#endif
