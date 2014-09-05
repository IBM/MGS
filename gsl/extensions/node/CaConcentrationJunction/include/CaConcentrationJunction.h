// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CaConcentrationJunction_H
#define CaConcentrationJunction_H

#include "Lens.h"
#include "CG_CaConcentrationJunction.h"
#include "rndm.h"

class CaConcentrationJunction : public CG_CaConcentrationJunction
{
   public:
      void initializeJunction(RNG& rng);
      void predictJunction(RNG& rng);
      void correctJunction(RNG& rng);
      virtual bool checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset);
      virtual bool confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset);
      virtual ~CaConcentrationJunction();
      float getArea(DimensionStruct* a, DimensionStruct* b);
      float getArea();
};

#endif
