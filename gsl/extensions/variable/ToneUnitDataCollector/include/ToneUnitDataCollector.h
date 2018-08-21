#ifndef ToneUnitDataCollector_H
#define ToneUnitDataCollector_H

#include "Lens.h"
#include "CG_ToneUnitDataCollector.h"
#include <memory>

class ToneUnitDataCollector : public CG_ToneUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ToneUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_ToneUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      ToneUnitDataCollector();
      virtual ~ToneUnitDataCollector();
      virtual void duplicate(std::auto_ptr<ToneUnitDataCollector>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ToneUnitDataCollector>& dup) const;
};

#endif
