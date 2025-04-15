// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConcentrationEndPoint_H
#define CaConcentrationEndPoint_H

#include "Lens.h"
#include "CG_CaConcentrationEndPoint.h"
#include "rndm.h"

class CaConcentrationEndPoint : public CG_CaConcentrationEndPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceSolvedCaConcentration(RNG& rng);
      void produceFinishedCaConcentration(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_CaConcentrationEndPointOutAttrPSet* CG_outAttrPset);
      virtual ~CaConcentrationEndPoint();
};

#endif
