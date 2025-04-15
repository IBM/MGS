// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NodeRelationalDataUnit_H
#define NodeRelationalDataUnit_H
#include "Copyright.h"

#include <deque>

class Constant;
class Edge;
class NodeDescriptor;
class VariableDescriptor;

class NodeRelationalDataUnit
{
   public:
      std::deque<Constant*>& getPreConstants() {
	 return _preConstants;
      }
      std::deque<Edge*>& getPreEdges() {
	 return _preEdges;
      }
      std::deque<NodeDescriptor*>& getPreNodes() {
	 return _preNodes;
      }
      std::deque<VariableDescriptor*>& getPreVariables() {
	 return _preVariables;
      }
      std::deque<Edge*>& getPostEdges() {
	 return _postEdges;
      }
      std::deque<NodeDescriptor*>& getPostNodes() {
	 return _postNodes;
      }
      std::deque<VariableDescriptor*>& getPostVariables() {
	 return _postVariables;
      }

   private:
      std::deque<Constant*> _preConstants;
      std::deque<Edge*> _preEdges;
      std::deque<NodeDescriptor*> _preNodes;
      std::deque<VariableDescriptor*> _preVariables;
      std::deque<Edge*> _postEdges;
      std::deque<NodeDescriptor*> _postNodes;
      std::deque<VariableDescriptor*> _postVariables;
};
#endif
