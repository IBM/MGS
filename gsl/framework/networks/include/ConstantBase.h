// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ConstantBase_H
#define ConstantBase_H
#include "Copyright.h"

#include "Constant.h"
#include "ConstantRelationalDataUnit.h"

#include <list>
#include <cassert>

class Edge;
class NodeDescriptor;
class VariableDescriptor;
class Simulation;
class Publisher;
class ConstantRelationalDataUnit;

class ConstantBase : public Constant
{

   public:
      ConstantBase(Simulation& sim);
      ConstantBase(const ConstantBase& rv);
      virtual const std::deque<Edge*>& getPostEdgeList();
      virtual const std::deque<NodeDescriptor*>& getPostNodeList();
      virtual const std::deque<VariableDescriptor*>& getPostVariableList();
      // virtual void store(std::ostream& os) = 0; // CG
      // virtual void reload(std::istream& is) = 0; // CG
      virtual ~ConstantBase();
      virtual Publisher* getPublisher() = 0;
      virtual Simulation& getSimulation() {return _sim;}

   protected:
      Publisher* _publisher;
      ConstantRelationalDataUnit* _relationalDataUnit;
      void checkAndAddPostEdge(Edge* e);
      void checkAndAddPostNode(NodeDescriptor* n);
      void checkAndAddPostVariable(VariableDescriptor* v);
      bool relationalDataEnabled();

   private:
      // disable due to reference
      ConstantBase& operator=(const ConstantBase& rv) {
	 assert(0);
	 return *this;
      }
      void copyOwnedHeap(const ConstantBase& rv);
      void destructOwnedHeap();
      Simulation& _sim;
};

#endif
