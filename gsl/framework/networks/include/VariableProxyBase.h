// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VariableProxyBase_H
#define VariableProxyBase_H
#include "Copyright.h"

#include "Variable.h"
#include "ConnectionIncrement.h"

#include <cassert>
#include <map>

class Constant;
class Edge;
class Node;
class Publisher;
class VariableDescriptor;

class VariableProxyBase : public Variable
{

   public:
      VariableProxyBase() : Variable(), _variableInstanceAccessor(0) {};
      virtual ~VariableProxyBase() {};

      virtual Variable* getVariable() {
         return _variableInstanceAccessor->getVariable();
      }

      virtual void setVariable(Variable* v) {
	_variableInstanceAccessor->setVariable(v);
      }

      virtual void setVariableDescriptor(VariableDescriptor* vd) {
	_variableInstanceAccessor=vd;
      }

      virtual void setVariableType(VariableCompCategoryBase* vcb) {
         _variableInstanceAccessor->setVariableType(vcb);
      }

      virtual VariableCompCategoryBase* getVariableType() {
         return _variableInstanceAccessor->getVariableType();
      }

      virtual int getVariableIndex() const {
         return _variableInstanceAccessor->getVariableIndex();
      }

      virtual void setVariableIndex(int index) {
         _variableInstanceAccessor->setVariableIndex(index);
      }

      virtual void initialize(ParameterSet* initPSet) {
	 assert(false);
      }

      virtual void acceptService(Service* service, const std::string& name) {
	 assert(false);
      }

      virtual Publisher* getPublisher() {
	 assert(false);
	 return 0;
      }

      virtual const char* getServiceName(void* data) const {
	 assert(false);
	 return "";
      }

      virtual const char* getServiceDescription(void* data) const {
	 assert(false);
	 return "";
      }

      virtual void addPreConstant(Constant* CG_constant, 
				  ParameterSet* CG_pset) {
	 assert(false);
      }

      virtual void addPreVariable(VariableDescriptor* CG_variable, 
				  ParameterSet* CG_pset) {
	 assert(false);
      }

      virtual void addPreEdge(Edge* CG_edge, ParameterSet* CG_pset) {
	 assert(false);
      }

      virtual void addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) {
	 assert(false);
      }

      /*
      virtual ConnectionIncrement* getComputeCost() {
         virtual std::map<std::string,ConnectionIncrement>* getComputeCost() {
	 return &_computeCost;
      }
      */

   protected:
      void checkAndAddPreConstant(Constant* c) {}
      void checkAndAddPreEdge(Edge* e) {}
      void checkAndAddPreNode(NodeDescriptor* n) {}
      void checkAndAddPreVariable(VariableDescriptor* v) {}
      void checkAndAddPostEdge(Edge* e) {}
      void checkAndAddPostNode(NodeDescriptor* n) {}
      void checkAndAddPostVariable(VariableDescriptor* v) {}

      virtual TriggerableBase::EventType createTriggerableCaller(
	 const std::string& name, NDPairList* ndpList, 
	 std::unique_ptr<TriggerableCaller>& triggerableCaller) {
	 throw SyntaxErrorException(
	    name + " is not defined in variable proxy.");
	 //return TriggerableBase::_UNALTERED;
      }
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args) {
	 assert(false);
      }
      virtual void doInitialize(const NDPairList& ndplist) {
	 assert(false);
      }
      ConnectionIncrement _computeCost;            // temporarilly added. 01/27/2006
//      std::map<std::string, ConnectionIncrement> _computeCost;            // temporarilly added. 01/27/2006


   private:
      VariableDescriptor* _variableInstanceAccessor;

};

#endif
