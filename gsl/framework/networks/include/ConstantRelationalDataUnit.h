// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
      void duplicate(std::auto_ptr<ConstantRelationalDataUnit>& dup) const {
	 dup.reset(new ConstantRelationalDataUnit(*this));
      }

   private:
      std::deque<Edge*> _postEdges;
      std::deque<NodeDescriptor*> _postNodes;
      std::deque<VariableDescriptor*> _postVariables;
};
#endif
