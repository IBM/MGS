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

#ifndef VoltageJunctionPoint_H
#define VoltageJunctionPoint_H

#include "Lens.h"
#include "CG_VoltageJunctionPoint.h"
#include "rndm.h"

class VoltageJunctionPoint : public CG_VoltageJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceVoltage(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageJunctionPointInAttrPSet* CG_inAttrPset, CG_VoltageJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~VoltageJunctionPoint();
};

#endif
