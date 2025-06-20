// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ConstantRelationalDataUnit_H
#define ConstantRelationalDataUnit_H
#include "Copyright.h"

#include <deque>
#include <memory>

class Edge;
class NodeDescriptor;
class VariableDescriptor;

class ConstantRelationalDataUnit
{
   public:
      std::deque<Edge*>& getPostEdges() {
	 return _postEdges;
      }
      std::deque<NodeDescriptor*>& getPostNodes() {
	 return _postNodes;
      }
      std::deque<VariableDescriptor*>& getPostVariables() {
	 return _postVariables;
      }
      void duplicate(std::unique_ptr<ConstantRelationalDataUnit>& dup) const {
	 dup.reset(new ConstantRelationalDataUnit(*this));
      }

   private:
      std::deque<Edge*> _postEdges;
      std::deque<NodeDescriptor*> _postNodes;
      std::deque<VariableDescriptor*> _postVariables;
};
#endif
