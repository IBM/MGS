// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConnexon_H
#define CaConnexon_H

#include "Lens.h"
#include "CG_CaConnexon.h"
#include "rndm.h"

class CaConnexon : public CG_CaConnexon
{
   public:
      void produceInitialState(RNG& rng);
      void produceState(RNG& rng);
      void computeState(RNG& rng);
      virtual void setCaPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConnexonInAttrPSet* CG_inAttrPset, CG_CaConnexonOutAttrPSet* CG_outAttrPset);
      virtual void setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConnexonInAttrPSet* CG_inAttrPset, CG_CaConnexonOutAttrPSet* CG_outAttrPset);
      virtual ~CaConnexon();
};

#endif
