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

#ifndef VariableBase_H
#define VariableBase_H
#include "Copyright.h"

#include "Variable.h"

#include <deque>
#include <cassert>

class Constant;
class Edge;
class NodeDescriptor;
class Simulation;
class Publisher;
class VariableRelationalDataUnit;

class VariableBase : public Variable
{

   public:
      VariableBase();
      VariableBase(const VariableBase& rv);
      virtual const std::deque<Constant*>& getPreConstantList();
      virtual const std::deque<Edge*>& getPreEdgeList();
      virtual const std::deque<NodeDescriptor*>& getPreNodeList();
      virtual const std::deque<VariableDescriptor*>& getPreVariableList();
      virtual const std::deque<Edge*>& getPostEdgeList();
      virtual const std::deque<NodeDescriptor*>& getPostNodeList();
      virtual const std::deque<VariableDescriptor*>& getPostVariableList();
      // virtual void store(std::ostream& os) = 0; // CG
      // virtual void reload(std::istream& is) = 0; // CG
      virtual ~VariableBase();
      virtual Publisher* getPublisher() = 0;

      virtual Variable* getVariable() {
         return this;
      }
      virtual void setVariableDescriptor(VariableDescriptor* vd) {
	_variableInstanceAccessor=vd;
      }
      virtual void setVariable(Variable* v) {
	_variableInstanceAccessor->setVariable(v);
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

      virtual bool hasService() {
         return true;
      }

   protected:
      Publisher* _publisher;
      void checkAndAddPreConstant(Constant* c);
      void checkAndAddPreEdge(Edge* e);
      void checkAndAddPreNode(NodeDescriptor* n);
      void checkAndAddPreVariable(VariableDescriptor* v);
      void checkAndAddPostEdge(Edge* e);
      void checkAndAddPostNode(NodeDescriptor* n);
      void checkAndAddPostVariable(VariableDescriptor* v);
      bool relationalDataEnabled();
      Simulation& getSimulation();

   private:
      VariableRelationalDataUnit* _relationalDataUnit;
      void copyOwnedHeap(const VariableBase& rv);
      void destructOwnedHeap();
      VariableDescriptor* _variableInstanceAccessor;

};

#endif
