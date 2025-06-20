// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EDGE_H
#define EDGE_H
#include "Copyright.h"

#include "Publishable.h"
#include "ServiceAcceptor.h"
#include "TriggerableBase.h"

#include <string>

class Constant;
class NodeDescriptor;
class VariableDescriptor;
class ParameterSet;

class Edge : public Publishable, public ServiceAcceptor,
	     public TriggerableBase
{
   public:
      virtual void initialize(ParameterSet* initPSet) = 0;
      virtual NodeDescriptor* getPreNode() = 0;
      virtual void setPreNode(NodeDescriptor*) = 0;
      virtual NodeDescriptor* getPostNode() = 0;
      virtual void setPostNode(NodeDescriptor*) = 0;
      virtual std::string getModelName() = 0;
      virtual ~Edge() {}

      virtual void getInitializationParameterSet(
	 std::unique_ptr<ParameterSet>&& r_aptr) const = 0;
      virtual void getInAttrParameterSet(
	 std::unique_ptr<ParameterSet>&& r_aptr) const = 0;
      virtual void getOutAttrParameterSet(
	 std::unique_ptr<ParameterSet>&& r_aptr) const = 0;

      virtual void addPreConstant(Constant* c, ParameterSet* InAttrPSet) = 0;
      virtual void addPreVariable(VariableDescriptor* v, ParameterSet* InAttrPSet) = 0;
      virtual void addPostVariable(VariableDescriptor* v, ParameterSet* OutAttrPSet) = 0;
};

#endif
