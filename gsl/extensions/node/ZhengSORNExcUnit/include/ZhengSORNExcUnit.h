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

#ifndef ZhengSORNExcUnit_H
#define ZhengSORNExcUnit_H

#include "Lens.h"
#include "CG_ZhengSORNExcUnit.h"
#include "rndm.h"
#include <fstream>

class ZhengSORNExcUnit : public CG_ZhengSORNExcUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void fire(RNG& rng);
      virtual void checkForSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void checkForInhSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void setE2EIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void setI2EIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual bool checkInitWeights(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNExcUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNExcUnitOutAttrPSet* CG_outAttrPset);
      virtual void outputWeights(std::ofstream& fsE2E, std::ofstream& fsI2E);
      virtual void inputWeights(int col, float weight);
      virtual void inputI2EWeights(int col, float weight);
      void inputTE(float val);
      virtual ~ZhengSORNExcUnit();
};

#endif
