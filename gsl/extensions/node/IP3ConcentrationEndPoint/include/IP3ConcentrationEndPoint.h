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

#ifndef IP3ConcentrationEndPoint_H
#define IP3ConcentrationEndPoint_H

#include "Lens.h"
#include "CG_IP3ConcentrationEndPoint.h"
#include "rndm.h"

class IP3ConcentrationEndPoint : public CG_IP3ConcentrationEndPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceSolvedIP3Concentration(RNG& rng);
      void produceFinishedIP3Concentration(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationEndPointOutAttrPSet* CG_outAttrPset);
      virtual ~IP3ConcentrationEndPoint();
};

#endif
