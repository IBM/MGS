#ifndef DNNode_H
#define DNNode_H

#include "Lens.h"
#include "CG_DNNode.h"
#include "rndm.h"

class DNNode : public CG_DNNode
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void extractInputIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DNNodeInAttrPSet* CG_inAttrPset, CG_DNNodeOutAttrPSet* CG_outAttrPset);
      virtual ~DNNode();
};

#endif
