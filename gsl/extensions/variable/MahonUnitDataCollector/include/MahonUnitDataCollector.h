#ifndef MahonUnitDataCollector_H
#define MahonUnitDataCollector_H

#include "Lens.h"
#include "CG_MahonUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class MahonUnitDataCollector : public CG_MahonUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void dataCollectionSpike(Trigger* trigger, NDPairList* ndPairList);
      virtual void dataCollectionLFP(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MahonUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MahonUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      MahonUnitDataCollector();
      virtual ~MahonUnitDataCollector();
      virtual void duplicate(std::auto_ptr<MahonUnitDataCollector>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_MahonUnitDataCollector>& dup) const;


 private:
      std::ofstream* spike_file;
      std::ofstream* x1_file;
      std::ofstream* x2_file;
      std::ofstream* x3_file;

};

#endif
