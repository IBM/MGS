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

#ifndef FSIIAFUnitDataCollector_H
#define FSIIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_FSIIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class FSIIAFUnitDataCollector : public CG_FSIIAFUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollectionSpikes(Trigger* trigger, NDPairList* ndPairList);
      virtual void dataCollectionOther(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_TraubIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_TraubIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      TraubIAFUnitDataCollector();
      virtual ~TraubIAFUnitDataCollector();
      virtual void duplicate(std::unique_ptr<TraubIAFUnitDataCollector>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_TraubIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* spikes_file;
};

#endif
