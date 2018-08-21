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

#ifndef VoltageEndPoint_H
#define VoltageEndPoint_H

#include "Lens.h"
#include "CG_VoltageEndPoint.h"
#include "rndm.h"

class VoltageEndPoint : public CG_VoltageEndPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceSolvedVoltage(RNG& rng);
      void produceFinishedVoltage(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageEndPointInAttrPSet* CG_inAttrPset, CG_VoltageEndPointOutAttrPSet* CG_outAttrPset);
      virtual ~VoltageEndPoint();
};

#endif
