// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Constant_H
#define Constant_H
//#include "Copyright.h"

#include <memory>
#include <vector>
#include "Publishable.h"

class DataItem;
class Edge;
class LensContext;
class NodeDescriptor;
class ParameterSet;
//class Variable;
class VariableDescriptor;
class NDPairList;

class Constant : public Publishable
{

   public:
      virtual void addPostEdge(Edge* e, ParameterSet* OutAttrPSet) = 0;
      virtual void addPostNode(NodeDescriptor* n, 
			       ParameterSet* OutAttrPSet) = 0;
      virtual void addPostVariable(VariableDescriptor* v, ParameterSet* OutAttrPSet) = 0;
      virtual ~Constant();
      virtual void getOutAttrParameterSet(
	 std::unique_ptr<ParameterSet>& outAttrPSet) const = 0;
      virtual void duplicate(std::unique_ptr<Constant>&& dup) const = 0;
      void initialize(LensContext *c, const std::vector<DataItem*>& args);
      void initialize(const NDPairList& ndplist);
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args) = 0;
      virtual void doInitialize(const NDPairList& ndplist) = 0;
};

#endif
