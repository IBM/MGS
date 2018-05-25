// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef FSIIAFUnit_H
#define FSIIAFUnit_H

#include "Lens.h"
#include "CG_FSIIAFUnit.h"
#include "rndm.h"
#include <fstream>

class FSIIAFUnit : public CG_FSIIAFUnit
{
   public:
      void initialize(RNG& rng);
      void updateInput(RNG& rng);
      void updateV(RNG& rng);
      void threshold(RNG& rng);
      void outputPSPs(std::ofstream& fs);
      void outputWeights(std::ofstream& fs);
      void outputGJs(std::ofstream& fs);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual bool bidirectional1(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual bool bidirectional2(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~FSIIAFUnit();
};

#endif
