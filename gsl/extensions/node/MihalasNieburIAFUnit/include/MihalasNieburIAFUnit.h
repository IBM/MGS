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
      void updateInput(RNG& rng);
      void updateV(RNG& rng);
      void threshold(RNG& rng);
      void outputWeights(std::ofstream& fs);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburIAFUnitInAttrPSet* CG_inAttrPset, CG_MihalasNieburIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~MihalasNieburIAFUnit();
};

#endif
