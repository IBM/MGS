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

#ifndef CaConcentrationJunctionPoint_H
#define CaConcentrationJunctionPoint_H

#include "Lens.h"
#include "CG_CaConcentrationJunctionPoint.h"
#include "rndm.h"

class CaConcentrationJunctionPoint : public CG_CaConcentrationJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceCaConcentration(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~CaConcentrationJunctionPoint();
};

#endif
