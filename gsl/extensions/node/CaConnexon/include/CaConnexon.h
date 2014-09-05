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
