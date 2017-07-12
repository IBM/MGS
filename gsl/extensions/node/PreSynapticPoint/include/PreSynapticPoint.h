// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef PreSynapticPoint_H
#define PreSynapticPoint_H

#include "Lens.h"
#include "CG_PreSynapticPoint.h"
#include "rndm.h"

class PreSynapticPoint : public CG_PreSynapticPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceState(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_PreSynapticPointInAttrPSet* CG_inAttrPset, CG_PreSynapticPointOutAttrPSet* CG_outAttrPset);
      virtual ~PreSynapticPoint();
};

#endif
