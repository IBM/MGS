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

#ifndef TraubIAFUnitDataCollector_H
#define TraubIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_TraubIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class TraubIAFUnitDataCollector : public CG_TraubIAFUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollectionSpikes(Trigger* trigger, NDPairList* ndPairList);
      virtual void dataCollectionOther(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_TraubIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_TraubIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      TraubIAFUnitDataCollector();
      virtual ~TraubIAFUnitDataCollector();
      virtual void duplicate(std::auto_ptr<TraubIAFUnitDataCollector>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_TraubIAFUnitDataCollector>& dup) const;
 private:
      std::ofstream* voltage_file;
      std::ofstream* threshold_file;
      std::ofstream* spike_file;
      std::ofstream* spikevoltage_file;
      std::ofstream* LFP_synapses_file;      
};

#endif
