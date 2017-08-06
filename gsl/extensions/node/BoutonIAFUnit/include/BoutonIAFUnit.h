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
      virtual void setSpikeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual void setECBIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~BoutonIAFUnit();
};

#endif
