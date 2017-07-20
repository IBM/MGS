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

#include "NodeInstanceAccessor.h"
//#include "CompCategory.h"
//#include "Simulation.h"
#include "Grid.h"
#include "GridLayerData.h"
#include "GridLayerDescriptor.h"
#include "NodeCompCategoryBase.h"

NodeInstanceAccessor::NodeInstanceAccessor()
   : _gridLayerData(0), _node(0), _nodeIndex(0), _index(0)
{
}

GridLayerData* NodeInstanceAccessor::getGridLayerData() const
{
   return _gridLayerData;
}

GridLayerDescriptor* NodeInstanceAccessor::getGridLayerDescriptor() const
{
   return _gridLayerData->getGridLayerDescriptor();
}

void NodeInstanceAccessor::setGridLayerData(GridLayerData* gridLayerData)
{
   _gridLayerData=gridLayerData;
}

int NodeInstanceAccessor::getDensityIndex() const
{
   int uniformDensity = _gridLayerData->getGridLayerDescriptor()->isUniform();
   if (uniformDensity == 1) {
      return 0;
   } else if (uniformDensity > 1) {
      return getIndex() - getNodeIndex() * uniformDensity;
   } else {
      return getIndex() - _gridLayerData->getNodeOffsets()[getNodeIndex()];
   }
}

void NodeInstanceAccessor::getNodeCoords(std::vector<int> & coords) const 
{
   _gridLayerData->getGridLayerDescriptor()->
      getGrid()->getNodeCoords(getNodeIndex(), coords);
}

void NodeInstanceAccessor::getNodeCoords(ShallowArray<unsigned, 3, 2>& coords) const 
{
   _gridLayerData->getGridLayerDescriptor()->
      getGrid()->getNodeCoords(getNodeIndex(), coords);
}

void NodeInstanceAccessor::getNodeCoords2Dim(int& x, int& y) const 
{
  _gridLayerData->getGridLayerDescriptor()->
     getGrid()->getNodeCoords(getNodeIndex(), x, y);
}

Node* NodeInstanceAccessor::getNode()
{
   return _node;
}

void NodeInstanceAccessor::setNode(Node* n)
{
   _node=n;
}

int NodeInstanceAccessor::getNodeIndex() const {
  return _nodeIndex;
}      

void NodeInstanceAccessor::setNodeIndex(int pos) { 
  _nodeIndex = pos;
}

int  NodeInstanceAccessor::getIndex() const {
  return _index;
}      
  
void NodeInstanceAccessor::setIndex(int pos) { 
  _index = pos;
}

int NodeInstanceAccessor::getGlobalIndex() const
{
  return getGridLayerData()->getNodeCompCategoryBase()->
    getGridLayerDataOffsets()[getGridLayerData()->getGridLayerIndex()] +
    getNodeIndex();
}

NodeInstanceAccessor::~NodeInstanceAccessor()
{
}
