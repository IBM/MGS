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

#ifndef RabinovichWinnerlessUnitDataCollector_H
#define RabinovichWinnerlessUnitDataCollector_H

#include "Lens.h"
#include "CG_RabinovichWinnerlessUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class RabinovichWinnerlessUnitDataCollector : public CG_RabinovichWinnerlessUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      RabinovichWinnerlessUnitDataCollector();
      virtual ~RabinovichWinnerlessUnitDataCollector();
      virtual void duplicate(std::auto_ptr<RabinovichWinnerlessUnitDataCollector>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_RabinovichWinnerlessUnitDataCollector>& dup) const;
 private:
      std::ofstream *X_file;
};

#endif
