// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
