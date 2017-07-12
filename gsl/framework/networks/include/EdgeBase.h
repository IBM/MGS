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
