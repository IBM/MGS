// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EdgeRelationalDataUnit_H
#define EdgeRelationalDataUnit_H
#include "Copyright.h"

#include <deque>
#include <memory>

class Constant;
class Edge;
class NodeDescriptor;
class VariableDescriptor;

class EdgeRelationalDataUnit
{
   public:
      EdgeRelationalDataUnit() : _preNode(0) {}

      std::deque<Constant*>& getPreConstants() {
	 return _preConstants;
      }
      NodeDescriptor* getPreNode() {
	 return _preNode;
      }
      std::deque<VariableDescriptor*>& getPreVariables() {
	 return _preVariables;
      }
      std::deque<VariableDescriptor*>& getPostVariables() {
	 return _postVariables;
      }
      void setPreNode(NodeDescriptor* n) {
	 _preNode = n;
      }
      void duplicate(std::unique_ptr<EdgeRelationalDataUnit>& dup) const {
	 dup.reset(new EdgeRelationalDataUnit(*this));
      }

   private:
      std::deque<Constant*> _preConstants;
      NodeDescriptor* _preNode;
      std::deque<VariableDescriptor*> _preVariables;
      std::deque<VariableDescriptor*> _postVariables;
};
#endif
