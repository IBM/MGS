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
