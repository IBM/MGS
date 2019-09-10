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

#ifndef NazeSORNInhUnit_H
#define NazeSORNInhUnit_H

#include "Lens.h"
#include "CG_NazeSORNInhUnit.h"
#include "rndm.h"
#include <fstream>

class NazeSORNInhUnit : public CG_NazeSORNInhUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void fire(RNG& rng);
      virtual void setExcIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNInhUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNInhUnitOutAttrPSet* CG_outAttrPset);
      virtual void setInhIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNInhUnitInAttrPSet* CG_inAttrPset, CG_NazeSORNInhUnitOutAttrPSet* CG_outAttrPset);
      void inputWeights(std::ifstream& fsE2I, int col, float weight);
      void outputWeights(std::ofstream& fsE2I);
      void inputTI(float val);
      virtual ~NazeSORNInhUnit();
};

#endif
