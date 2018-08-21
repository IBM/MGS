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
      void duplicate(std::auto_ptr<EdgeRelationalDataUnit>& dup) const {
	 dup.reset(new EdgeRelationalDataUnit(*this));
      }

   private:
      std::deque<Constant*> _preConstants;
      NodeDescriptor* _preNode;
      std::deque<VariableDescriptor*> _preVariables;
      std::deque<VariableDescriptor*> _postVariables;
};
#endif
