#ifndef LypCollector_H
#define LypCollector_H

#include "Lens.h"
#include "CG_LypCollector.h"
#include "rndm.h"

class LypCollector : public CG_LypCollector
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LypCollectorInAttrPSet* CG_inAttrPset, CG_LypCollectorOutAttrPSet* CG_outAttrPset);
      virtual ~LypCollector();
};

#endif
