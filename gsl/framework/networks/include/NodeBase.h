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

#ifndef NodeBase_H
#define NodeBase_H
#include "Copyright.h"

#include "Node.h"
#include "NodeRelationalDataUnit.h"
#include "ShallowArray.h"

#include <deque>
#include <vector>
#include <cassert>

class Constant;
class Edge;
class Variable;
class VariableDescriptor;
class NodeDescriptor;
class GridLayerData;
class GridLayerDescriptor;
class Publisher;
class Simulation;

class NodeBase : public Node
{

   public:
      NodeBase();
      virtual const std::deque<Constant*>& getPreConstantList();
      virtual const std::deque<Edge*>& getPreEdgeList();
      virtual const std::deque<NodeDescriptor*>& getPreNodeList();
      virtual const std::deque<VariableDescriptor*>& getPreVariableList();
      virtual const std::deque<Edge*>& getPostEdgeList();
      virtual const std::deque<NodeDescriptor*>& getPostNodeList();
      virtual const std::deque<VariableDescriptor*>& getPostVariableList();
      virtual Simulation& getSimulation();

      // Node Descriptor implementation functions.
      virtual GridLayerData* getGridLayerData() const;
      virtual GridLayerDescriptor* getGridLayerDescriptor() const;
      virtual void setGridLayerData(GridLayerData* gridLayerData);
      virtual int getDensityIndex() const;
      virtual void getNodeCoords(std::vector<int> & coords) const;
      virtual void getNodeCoords(ShallowArray<unsigned, 3, 2>& coords) const;
      virtual void getNodeCoords2Dim(int& x, int& y) const;
      virtual Node* getNode() {
	 return this;
      }
      virtual void setNode(Node*);
      virtual int getNodeIndex() const;
      virtual void setNodeIndex(int pos);
      virtual int getIndex() const;
      virtual void setIndex(int pos);
      virtual int getGlobalIndex() const;

      virtual void setNodeDescriptor(NodeDescriptor* nodeDescriptor) {
	_nodeInstanceAccessor = nodeDescriptor;
      }
      virtual NodeDescriptor* getNodeDescriptor() const {return _nodeInstanceAccessor;}
      virtual ~NodeBase();
      virtual Publisher* getPublisher() = 0;

      virtual bool hasService() {
         return true;
      }

   protected:
      Publisher* _publisher;
      NodeRelationalDataUnit* _relationalInformation;      

      void checkAndAddPreConstant(Constant* c);
      void checkAndAddPreEdge(Edge* e);
      void checkAndAddPreNode(NodeDescriptor* n);
      void checkAndAddPreVariable(VariableDescriptor* v);
      void checkAndAddPostEdge(Edge* e);
      void checkAndAddPostNode(NodeDescriptor* n);
      void checkAndAddPostVariable(VariableDescriptor* v);
      inline bool relationalDataEnabled();      

   private:
      NodeDescriptor* _nodeInstanceAccessor;
//      ConnectionIncrement& _connectionIncrement;
};

#endif
