// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef IP3ConcentrationJunctionPoint_H
#define IP3ConcentrationJunctionPoint_H

#include "Lens.h"
#include "CG_IP3ConcentrationJunctionPoint.h"
#include "rndm.h"

class IP3ConcentrationJunctionPoint : public CG_IP3ConcentrationJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceIP3Concentration(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~IP3ConcentrationJunctionPoint();
};

#endif
