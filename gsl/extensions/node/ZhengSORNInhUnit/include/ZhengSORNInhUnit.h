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

#ifndef ZhengSORNInhUnit_H
#define ZhengSORNInhUnit_H

#include "Lens.h"
#include "CG_ZhengSORNInhUnit.h"
#include "rndm.h"
#include <fstream>

class ZhengSORNInhUnit : public CG_ZhengSORNInhUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void fire(RNG& rng);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNInhUnitInAttrPSet* CG_inAttrPset, CG_ZhengSORNInhUnitOutAttrPSet* CG_outAttrPset);
      void inputWeights(std::ifstream& fsE2I, int col, float weight);
      void outputWeights(std::ofstream& fsE2I);
      void inputTI(float val);
      virtual ~ZhengSORNInhUnit();
};

#endif
