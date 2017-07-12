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

#ifndef CaConcentrationEndPoint_H
#define CaConcentrationEndPoint_H

#include "Lens.h"
#include "CG_CaConcentrationEndPoint.h"
#include "rndm.h"

class CaConcentrationEndPoint : public CG_CaConcentrationEndPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceSolvedCaConcentration(RNG& rng);
      void produceFinishedCaConcentration(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_CaConcentrationEndPointOutAttrPSet* CG_outAttrPset);
      virtual ~CaConcentrationEndPoint();
};

#endif
