// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ZhengSORNExcUnit_H
#define ZhengSORNExcUnit_H

#include "Mgs.h"
#include "CG_ZhengSORNExcUnit.h"
#include "rndm.h"
#include <fstream>

class ZhengSORNExcUnit : public CG_ZhengSORNExcUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void fire(RNG& rng);
      virtual void checkForSynapse(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void checkForInhSynapse(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void setE2EIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void setI2EIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual bool checkInitWeights(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void outputWeights(std::ofstream& fsE2E, std::ofstream& fsI2E);
      virtual void inputWeights(int col, float weight);
      virtual void inputI2EWeights(int col, float weight);
      void inputTE(float val);
      virtual ~ZhengSORNExcUnit();
};

#endif
