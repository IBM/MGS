// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IP3ConcentrationEndPoint_H
#define IP3ConcentrationEndPoint_H

#include "Mgs.h"
#include "CG_IP3ConcentrationEndPoint.h"
#include "rndm.h"

class IP3ConcentrationEndPoint : public CG_IP3ConcentrationEndPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceSolvedIP3Concentration(RNG& rng);
      void produceFinishedIP3Concentration(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationEndPointOutAttrPSet* CG_outAttrPset);
      virtual ~IP3ConcentrationEndPoint();
};

#endif
