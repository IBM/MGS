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

#include "NodeProxyBase.h"
#include "Simulation.h"
#include "GridLayerData.h"
#include "GridLayerDescriptor.h"
#include "SyntaxErrorException.h"
#include "Grid.h"
#include "NodeCompCategoryBase.h"

NodeProxyBase::NodeProxyBase() 
   : Node(),  _relationalInformation(0), _nodeInstanceAccessor(0), _partition(0)
{
}

NodeProxyBase::~NodeProxyBase() 
{
  delete _relationalInformation;
}

TriggerableBase::EventType NodeProxyBase::createTriggerableCaller(
   const std::string& name, NDPairList* ndpList, 
   std::unique_ptr<TriggerableCaller>&& triggerableCaller) {
   throw SyntaxErrorException(
      name + " is not defined in node proxy.");
   //return TriggerableBase::_UNALTERED;
}

GridLayerData* NodeProxyBase::getGridLayerData() const
{
   return _nodeInstanceAccessor->getGridLayerData(); 
}

GridLayerDescriptor* NodeProxyBase::getGridLayerDescriptor() const
{
   return _nodeInstanceAccessor->getGridLayerData()->getGridLayerDescriptor(); 
}

void NodeProxyBase::setGridLayerData(GridLayerData* gridLayerData)
{
   _nodeInstanceAccessor->setGridLayerData(gridLayerData);
}

int NodeProxyBase::getDensityIndex() const
{
  return _nodeInstanceAccessor->getDensityIndex();
}

void NodeProxyBase::getNodeCoords(std::vector<int> & coords) const 
{
  _nodeInstanceAccessor->getNodeCoords(coords);
}

void NodeProxyBase::getNodeCoords(ShallowArray<unsigned, 3, 2>& coords) const 
{
  _nodeInstanceAccessor->getNodeCoords(coords);
}

void NodeProxyBase::getNodeCoords2Dim(int& x, int& y) const 
{
  _nodeInstanceAccessor->getNodeCoords2Dim(x, y);
}

void NodeProxyBase::setNode(Node* node) 
{
  _nodeInstanceAccessor->setNode(node);
}

int NodeProxyBase::getNodeIndex() const 
{
  return _nodeInstanceAccessor->getNodeIndex();
}      

void NodeProxyBase::setNodeIndex(int nodeIndex) 
{
  _nodeInstanceAccessor->setNodeIndex(nodeIndex);
}

int  NodeProxyBase::getIndex() const 
{
  return _nodeInstanceAccessor->getIndex();
}      
  
void NodeProxyBase::setIndex(int index)
{
  _nodeInstanceAccessor->setIndex(index);
}

int NodeProxyBase::getGlobalIndex() const
{
  return _nodeInstanceAccessor->getGridLayerData()->getNodeCompCategoryBase()->
    getGridLayerDataOffsets()[_nodeInstanceAccessor->getGridLayerData()->getGridLayerIndex()] +
    getNodeIndex();
}
                                                                                             
void NodeProxyBase::checkAndAddPreConstant(Constant* c) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreConstants().push_back(c);
   }
}
                                                                                                      
void NodeProxyBase::checkAndAddPreEdge(Edge* e) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreEdges().push_back(e);
   }
}
                                                                                                      
void NodeProxyBase::checkAndAddPreNode(NodeDescriptor* n) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreNodes().push_back(n);
   }
}
                                                                                                      
void NodeProxyBase::checkAndAddPreVariable(VariableDescriptor* v) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPreVariables().push_back(v);
   }
}
                                                                                                      
void NodeProxyBase::checkAndAddPostEdge(Edge* e) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPostEdges().push_back(e);
   }
}
                                                                                                      
void NodeProxyBase::checkAndAddPostNode(NodeDescriptor* n) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPostNodes().push_back(n);
   }
}
                                                                                                      
void NodeProxyBase::checkAndAddPostVariable(VariableDescriptor* v) {
   if (relationalDataEnabled()) {
      _relationalInformation->getPostVariables().push_back(v);
   }
}
    
bool NodeProxyBase::relationalDataEnabled() {
   bool rval=false;
   if ( _nodeInstanceAccessor->getGridLayerData()->getNodeCompCategoryBase()->
	getSimulation().isEdgeRelationalDataEnabled()) {
     if (_relationalInformation==0) _relationalInformation=new NodeRelationalDataUnit;
     rval=true;
   }
  return rval;
}

const std::deque<Constant*>& NodeProxyBase::getPreConstantList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreConstantList called when RelationalData is disabled!"
               << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreConstants();
}

const std::deque<Edge*>& NodeProxyBase::getPreEdgeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreEdgeList called when RelationalData is disabled!"
               << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreEdges();
}

const std::deque<NodeDescriptor*>& NodeProxyBase::getPreNodeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreNodeList called when RelationalData is disabled!"
               << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreNodes();
}

const std::deque<VariableDescriptor*>& NodeProxyBase::getPreVariableList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPreVariableList called when RelationalData is disabled!"
               << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPreVariables();
}

const std::deque<Edge*>& NodeProxyBase::getPostEdgeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPostEdgeList called when RelationalData is disabled!"
               << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPostEdges();
}
         
const std::deque<NodeDescriptor*>& NodeProxyBase::getPostNodeList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPostNodeList called when RelationalData is disabled!"
               << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPostNodes();
}
         
const std::deque<VariableDescriptor*>& NodeProxyBase::getPostVariableList()
{
   if (!relationalDataEnabled()) {
      // @TODO: Use exceptions
      std::cerr<<"getPostVariableList called when RelationalData is disabled!"
               << std::endl;
      exit(-1);
   }
   return _relationalInformation->getPostVariables();
}

