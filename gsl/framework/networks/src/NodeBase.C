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

#include "NodeBase.h"
#include "GridLayerData.h"
#include "NodeCompCategoryBase.h"
#include "Publisher.h"
#include "Simulation.h"
//#include "Grid.h"

// For now
//#include <iostream>

NodeBase::NodeBase()
   : Node(), _publisher(0),  _relationalInformation(0), _nodeInstanceAccessor(0)
{
}

void NodeBase::checkAndAddPreConstant(Constant* c) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreConstants().push_back(c);
   }
}

void NodeBase::checkAndAddPreEdge(Edge* e) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreEdges().push_back(e);
   }
}

void NodeBase::checkAndAddPreNode(NodeDescriptor* n) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreNodes().push_back(n);
   }
}

void NodeBase::checkAndAddPreVariable(VariableDescriptor* v) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreVariables().push_back(v);
   }
}

void NodeBase::checkAndAddPostEdge(Edge* e) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPostEdges().push_back(e);
   }
}

void NodeBase::checkAndAddPostNode(NodeDescriptor* n) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPostNodes().push_back(n);
   }
}

void NodeBase::checkAndAddPostVariable(VariableDescriptor* v) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPostVariables().push_back(v);
   }
}

bool NodeBase::relationalDataEnabled() {
   bool rval=false;
   if (_nodeInstanceAccessor->getGridLayerData()->getNodeCompCategoryBase()->
       getSimulation().isEdgeRelationalDataEnabled()) {
     if (_relationalInformation==0) _relationalInformation=new NodeRelationalDataUnit;
     rval=true;
   }
   return rval;
}

const std::deque<Constant*>& NodeBase::getPreConstantList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreConstantList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreConstants();

}

const std::deque<Edge*>& NodeBase::getPreEdgeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreEdgeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreEdges();

}

const std::deque<NodeDescriptor*>& NodeBase::getPreNodeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreNodes();

}

const std::deque<VariableDescriptor*>& NodeBase::getPreVariableList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreVariableList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreVariables();

}

const std::deque<Edge*>& NodeBase::getPostEdgeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPostEdgeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPostEdges();
}

const std::deque<NodeDescriptor*>& NodeBase::getPostNodeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPostNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPostNodes();
}

const std::deque<VariableDescriptor*>& NodeBase::getPostVariableList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPostVariableList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPostVariables();
}

Simulation& NodeBase::getSimulation() 
{
  return _nodeInstanceAccessor->getGridLayerData()->getNodeCompCategoryBase()->getSimulation();
}

GridLayerData* NodeBase::getGridLayerData() const
{
   return _nodeInstanceAccessor->getGridLayerData(); 
}

GridLayerDescriptor* NodeBase::getGridLayerDescriptor() const
{
   return _nodeInstanceAccessor->getGridLayerData()->getGridLayerDescriptor(); 
}

void NodeBase::setGridLayerData(GridLayerData* gridLayerData)
{
   _nodeInstanceAccessor->setGridLayerData(gridLayerData);
}

int NodeBase::getDensityIndex() const
{
  return _nodeInstanceAccessor->getDensityIndex();
}

void NodeBase::getNodeCoords(std::vector<int> & coords) const 
{
  _nodeInstanceAccessor->getNodeCoords(coords);
}

void NodeBase::getNodeCoords(ShallowArray<unsigned, 3, 2>& coords) const 
{
  _nodeInstanceAccessor->getNodeCoords(coords);
}

void NodeBase::getNodeCoords2Dim(int& x, int& y) const 
{
  _nodeInstanceAccessor->getNodeCoords2Dim(x, y);
}

void NodeBase::setNode(Node* node) 
{
  _nodeInstanceAccessor->setNode(node);
}

int NodeBase::getNodeIndex() const 
{
  return _nodeInstanceAccessor->getNodeIndex();
}      

void NodeBase::setNodeIndex(int nodeIndex) 
{
  _nodeInstanceAccessor->setNodeIndex(nodeIndex);
}

int  NodeBase::getIndex() const 
{
  return _nodeInstanceAccessor->getIndex();
}      
  
void NodeBase::setIndex(int index)
{
  _nodeInstanceAccessor->setIndex(index);
}

int NodeBase::getGlobalIndex() const
{
  return _nodeInstanceAccessor->getGridLayerData()->getNodeCompCategoryBase()->
    getGridLayerDataOffsets()[_nodeInstanceAccessor->getGridLayerData()->getGridLayerIndex()] +
    getNodeIndex();
}

NodeBase::~NodeBase()
{
   delete _publisher;
   delete _relationalInformation;
}
