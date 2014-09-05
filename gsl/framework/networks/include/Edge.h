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
	 std::auto_ptr<ParameterSet> & r_aptr) const = 0;
      virtual void getInAttrParameterSet(
	 std::auto_ptr<ParameterSet> & r_aptr) const = 0;
      virtual void getOutAttrParameterSet(
	 std::auto_ptr<ParameterSet> & r_aptr) const = 0;

      virtual void addPreConstant(Constant* c, ParameterSet* InAttrPSet) = 0;
      virtual void addPreVariable(VariableDescriptor* v, ParameterSet* InAttrPSet) = 0;
      virtual void addPostVariable(VariableDescriptor* v, ParameterSet* OutAttrPSet) = 0;
};

#endif
