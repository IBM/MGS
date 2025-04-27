// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Connexon_H
#define Connexon_H

#include "Lens.h"
#include "CG_Connexon.h"
#include "rndm.h"

class Connexon : public CG_Connexon
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ConnexonInAttrPSet* CG_inAttrPset, CG_ConnexonOutAttrPSet* CG_outAttrPset);
      virtual ~Connexon();
};

#endif
