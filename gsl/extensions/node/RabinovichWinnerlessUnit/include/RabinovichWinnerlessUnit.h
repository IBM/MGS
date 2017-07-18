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

#ifndef RabinovichWinnerlessUnit_H
#define RabinovichWinnerlessUnit_H

#include "Lens.h"
#include "CG_RabinovichWinnerlessUnit.h"
#include "rndm.h"
#include <fstream>

class RabinovichWinnerlessUnit : public CG_RabinovichWinnerlessUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void copy(RNG& rng);
      void outputWeights(std::ofstream& fsLN, std::ofstream& fsDR, std::ofstream& fsNS);
      virtual void checkForCorticalSynapse(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual void setLateralIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual void setModulatoryIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual void assymetric(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitOutAttrPSet* CG_outAttrPset);
      virtual ~RabinovichWinnerlessUnit();
};

#endif
