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

#ifndef MihalasNieburIAFUnitDataCollector_H
#define MihalasNieburIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_MihalasNieburIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class MihalasNieburIAFUnitDataCollector : public CG_MihalasNieburIAFUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MihalasNieburIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      MihalasNieburIAFUnitDataCollector();
      virtual ~MihalasNieburIAFUnitDataCollector();
      virtual void duplicate(std::unique_ptr<MihalasNieburIAFUnitDataCollector>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_MihalasNieburIAFUnitDataCollector>& dup) const;
 private:
      std::ofstream* voltage_file;
      std::ofstream* threshold_file;
      std::ofstream* spike_file;
};

#endif
