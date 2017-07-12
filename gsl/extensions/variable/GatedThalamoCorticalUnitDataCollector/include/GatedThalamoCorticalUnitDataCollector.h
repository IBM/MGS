#ifndef GatedThalamoCorticalUnitDataCollector_H
#define GatedThalamoCorticalUnitDataCollector_H

#include "Lens.h"
#include "CG_GatedThalamoCorticalUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class GatedThalamoCorticalUnitDataCollector : public CG_GatedThalamoCorticalUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamoCorticalUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_GatedThalamoCorticalUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      GatedThalamoCorticalUnitDataCollector();
      virtual ~GatedThalamoCorticalUnitDataCollector();
      virtual void duplicate(std::auto_ptr<GatedThalamoCorticalUnitDataCollector>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_GatedThalamoCorticalUnitDataCollector>& dup) const;

 private:
      std::ofstream* file;
      std::ofstream* yfile;
};

#endif
