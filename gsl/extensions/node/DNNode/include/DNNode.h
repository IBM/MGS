// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef DNNode_H
#define DNNode_H

#include "Mgs.h"
#include "CG_DNNode.h"
#include "rndm.h"

class DNNode : public CG_DNNode
{
   public:
      using Node::initialize;
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void extractInputIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_DNNodeInAttrPSet* CG_inAttrPset, CG_DNNodeOutAttrPSet* CG_outAttrPset);
      virtual ~DNNode();
};

#endif
