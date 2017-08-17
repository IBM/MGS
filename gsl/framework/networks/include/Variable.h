// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef Variable_H
#define Variable_H
#include "Copyright.h"

//#include "Publishable.h"
#include "ServiceAcceptor.h"
#include "TriggerableBase.h"
#include "VariableDescriptor.h"

#include <map>

class Constant;
class Edge;
class NodeDescriptor;
class DataItem;
class LensContext;
class ParameterSet;
class NDPairList;
class ConnectionIncrement;

class Variable : public VariableDescriptor, public ServiceAcceptor,
		 public TriggerableBase
{

   public:      
  Variable() : _compCategory(0) {}
      virtual void setVariableDescriptor(VariableDescriptor* vd) = 0;
      virtual void addPostEdge(Edge* e, ParameterSet* OutAttrPSet) = 0;
      virtual void addPostNode(NodeDescriptor* n, 
			       ParameterSet* OutAttrPSet) = 0;
      virtual void addPostVariable(VariableDescriptor* v, ParameterSet* OutAttrPSet) = 0;
      virtual void addPreConstant(Constant* c, ParameterSet* InAttrPSet) = 0;
      virtual void addPreEdge(Edge* e, ParameterSet* InAttrPSet) = 0;
      virtual void addPreNode(NodeDescriptor* n, ParameterSet* InAttrPSet) = 0;
      virtual void addPreVariable(VariableDescriptor* v, ParameterSet* InAttrPSet) = 0;
      virtual void getInitializationParameterSet(
	 std::auto_ptr<ParameterSet> & r_aptr) const = 0;
      virtual void getInAttrParameterSet(
	 std::auto_ptr<ParameterSet>& inAttrPSet) const = 0;
      virtual void getOutAttrParameterSet(
	 std::auto_ptr<ParameterSet>& outAttrPSet) const = 0;
      virtual ~Variable();
      virtual void duplicate(std::auto_ptr<Variable>& dup) const = 0;
      void initialize(LensContext *c, const std::vector<DataItem*>& args);
      void initialize(const NDPairList& ndplist);
//      virtual ConnectionIncrement* getComputeCost() const = 0;
//      virtual ConnectionIncrement* getComputeCost();
//      virtual std::map<std::string, ConnectionIncrement>* getComputeCost() = 0;

   protected:
      VariableCompCategoryBase* _compCategory;
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args) = 0;
      virtual void doInitialize(const NDPairList& ndplist) = 0;
};

#endif
