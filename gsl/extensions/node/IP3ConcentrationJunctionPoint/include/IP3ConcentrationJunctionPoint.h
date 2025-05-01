// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IP3ConcentrationJunctionPoint_H
#define IP3ConcentrationJunctionPoint_H

#include "Mgs.h"
#include "CG_IP3ConcentrationJunctionPoint.h"
#include "rndm.h"

class IP3ConcentrationJunctionPoint : public CG_IP3ConcentrationJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceIP3Concentration(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~IP3ConcentrationJunctionPoint();
};

#endif
