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

#ifndef VariableRelationalDataUnit_H
#define VariableRelationalDataUnit_H
#include "Copyright.h"

#include <deque>

class Constant;
class Edge;
class NodeDescriptor;
class Variable;

class VariableRelationalDataUnit
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
      void duplicate(std::unique_ptr<VariableRelationalDataUnit>& dup) const {
	 dup.reset(new VariableRelationalDataUnit(*this));
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
