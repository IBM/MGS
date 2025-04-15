// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EdgeBase_H
#define EdgeBase_H
#include "Copyright.h"

#include "Edge.h"
#include "Publisher.h"

#include <deque>
#include <vector>
#include <cassert>

class Constant;
class NodeDescriptor;
class VariableDescriptor;
class Simulation; 
class EdgeCompCategoryBase;
class EdgeRelationalDataUnit;
class Publisher;

class EdgeBase : public Edge
{

   public:
      //EdgeBase(EdgeCompCategoryBase* cc);
      EdgeBase();
      // virtual void initialize(ParameterSet* initPSet) =0; // CG
      // virtual void addPreEdge(Edge* e, ParameterSet* InAttrPSet)=0; // CG
      // virtual void addPostEdge(Edge* e, ParameterSet* OutAttrPSet)=0; // CG
      virtual const std::deque<Constant*>& getPreConstantList();
      virtual const std::deque<VariableDescriptor*>& getPreVariableList();
      virtual const std::deque<VariableDescriptor*>& getPostVariableList();
      virtual NodeDescriptor* getPreNode();
      virtual NodeDescriptor* getPostNode();
      EdgeBase(const EdgeBase& rv);
      // virtual void store(std::ostream& os) = 0; // CG
      // virtual void reload(std::istream& is) = 0; // CG
      virtual ~EdgeBase();
      virtual Publisher* getPublisher() = 0;
      Simulation& getSimulation();
      virtual std::string getModelName();
      void setEdgeCompCategoryBase(EdgeCompCategoryBase* cc);
     
   protected:
      Publisher* _publisher;
      EdgeRelationalDataUnit* _relationalDataUnit;
      void checkAndAddPreConstant(Constant* c);
      void checkAndAddPreVariable(VariableDescriptor* v);
      void checkAndAddPostVariable(VariableDescriptor* v);
      void checkAndSetPreNode(NodeDescriptor* n);
      void checkAndSetPostNode(NodeDescriptor* n);
      inline bool relationalDataEnabled();
   private:
      EdgeCompCategoryBase* _edgeCompCategoryBase;
      // Post Node is moved here because it should be available even though
      // the relational data is disabled, the granule of the Edge is the
      // same as its post node...
      NodeDescriptor* _postNode;
};

#endif
