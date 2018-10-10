#ifndef GatedThalamicUnitDataCollector_H
#define GatedThalamicUnitDataCollector_H

#include "Lens.h"
#include "CG_GatedThalamicUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class GatedThalamicUnitDataCollector : public CG_GatedThalamicUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamicUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_GatedThalamicUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      GatedThalamicUnitDataCollector();
      virtual ~GatedThalamicUnitDataCollector();
      virtual void duplicate(std::unique_ptr<GatedThalamicUnitDataCollector>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GatedThalamicUnitDataCollector>& dup) const;

 private:
      std::ofstream* file;
};

#endif
