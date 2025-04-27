// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BoutonIAFUnit_H
#define BoutonIAFUnit_H

#include "Lens.h"
#include "CG_BoutonIAFUnit.h"
#include "rndm.h"

class BoutonIAFUnit : public CG_BoutonIAFUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void outputIndexs(std::ofstream& fs);
      virtual void setSpikeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual void seteCBIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~BoutonIAFUnit();
};

#endif
