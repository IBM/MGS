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

#include "NodeSet.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "NodeDescriptor.h"
#include "VolumeOdometer.h"
#include "NodeAccessor.h"
#include "SyntaxErrorException.h"
#include <algorithm>

NodeSet::NodeSet(Grid* grid)
   : GridSet(grid), _allLayers(true), _allIndices(true)
{
}

NodeSet::NodeSet(NodeDescriptor* n)
   : GridSet(n->getGridLayerDescriptor()->getGrid()), _allLayers(false),
     _allIndices(false)
{
   _layers.push_back(n->getGridLayerDescriptor());
   _allCoords = false;
   n->getNodeCoords(_beginCoords);
   _endCoords = _beginCoords;
   _indices.push_back(n->getDensityIndex());
}

NodeSet::NodeSet(Grid* grid, std::vector<NodeDescriptor*> nodes)
  : GridSet(grid), _allLayers(false), _allIndices(false)
{
  _allCoords=false;
  sort(nodes.begin(), nodes.end());
  std::vector<NodeDescriptor*>::iterator newEnd=unique(nodes.begin(), nodes.end());
  nodes.erase(newEnd, nodes.end());
  std::vector<NodeDescriptor*>::iterator nsiter=nodes.begin(), nsend=nodes.end();
  if (nsiter!=nsend) {
    (*nsiter)->getNodeCoords(_beginCoords);
    _endCoords = _beginCoords;
  }
  for (; nsiter!=nsend; ++nsiter) {
    _indices.push_back((*nsiter)->getDensityIndex());
    std::vector<GridLayerDescriptor*>::iterator liter=_layers.begin(), lend=_layers.end();
    bool foundLayer=false;
    GridLayerDescriptor* gld=(*nsiter)->getGridLayerDescriptor();
    if (grid!=gld->getGrid()) {
      std::cerr<<"Nodes in a Node Set must derive from the same grid!"<<std::endl;
      exit(-1);
    }
    for (; liter!=lend; ++liter) if (gld== *liter) foundLayer=true;
    if (!foundLayer)_layers.push_back((*nsiter)->getGridLayerDescriptor());
    std::vector<int> coords;
    (*nsiter)->getNodeCoords(coords);
    if (grid->getNodeIndex(coords)!=grid->getNodeIndex(_beginCoords)) {
      std::cerr<<"Nodes in a Node Set must derive from the same grid coordinate!"<<std::endl;
      exit(-1);
    }
  }
}

NodeSet::NodeSet(const GridSet& gs)
   : GridSet(gs), _allLayers(true), _allIndices(true)
{
}

NodeSet::~NodeSet()
{
}

bool NodeSet::contains(NodeDescriptor* n)
{
   bool rval = true;
   GridLayerDescriptor* gld = n->getGridLayerDescriptor();

   // Check if node's GridLayerDescriptor is in NodeSet
   if (_allLayers) {
      if(gld->getGrid() != _grid) {
         rval = false;
      }
   } else {
      std::vector<GridLayerDescriptor*>::iterator it, end = _layers.end();
      for (it = _layers.begin(); it != end; ++it) {
         if ((*it) == gld) {
	    break;
	 }
      }
      if (it == end) {
	 rval = false;
      }
   }

   // Check if node's coordinate is in NodeSet coordinate range
   if (rval && !_allCoords) {
      std::vector<int> coords;
      int coord;
      n->getNodeCoords(coords);
      int end = coords.size();
      for (int i = 0; i < end; ++i) {
         coord = coords[i];
         if (coord < _beginCoords[i] || coord > _endCoords[i] || coord%_increment[i]) {
            rval = false;
            break;
         }
      }
   }

   // Check if node's index is in index range
   if (rval && !_allIndices) {
      int index = n->getDensityIndex();
      std::vector<int>::iterator it, end = _indices.end();
      for (it = _indices.begin(); it != end; ++it) {
         if ((*it) == index) break;
      }
      if (it == end) rval = false;
   }
   return rval;
}

void NodeSet::getNodes(std::vector<NodeDescriptor*>& nodes)
{
    /*
      Strategy:
      1) determine coordinate range, layers, and indices
      2) loop through coordinate range
      3) loop through layers
      rest in inlined function:
         4) loop through valid indices
         5) push Node * onto vector
   */

   nodes.clear();

   // make sure _layers contains valid layers
   resetAllLayers();
   std::vector<GridLayerDescriptor*>::const_iterator it, end = _layers.end();

   unsigned coordIndex;

   // main loop
   VolumeOdometer vo(_beginCoords, _increment, _endCoords);
   std::vector<int>& coords = vo.look();
   for (; !vo.isRolledOver(); vo.next()) {
      coordIndex = _grid->getNodeIndex(coords);
      for(it = _layers.begin(); it != end; ++it) {
	getNodesWithCoordinates(nodes, (*it), coordIndex);
      }
   }
}

void NodeSet::getNodes(std::vector<NodeDescriptor*>& nodes, 
		       std::string gldName)
{
   std::vector<GridLayerDescriptor*>::iterator it, end = _layers.end();
   GridLayerDescriptor* gld = 0;
   for (it = _layers.begin(); it != end; ++it) {
      if ((*it)->getName() == gldName) {
         gld = (*it);
         break;
      }
   }
   if (gld) {
      getNodes(nodes, gld);
   } else {
      throw SyntaxErrorException("NodeSet::getNodes: name argument does not match any GridLayerDescriptor name.");
   }
}

void NodeSet::getNodes(std::vector<NodeDescriptor*> &nodes, 
		       GridLayerDescriptor* gld)
{
   /*
      Strategy:
      1) determine coordinate range, and indices, and verify valid layer
      2) loop through coordinate range for given layer
      rest in inlined function:
         3) loop through valid indices
         4) push Node * onto vector
   */

   // make sure _layers contains valid layers
   resetAllLayers();

   nodes.clear();

   // make sure specified gld is valid
   std::vector<GridLayerDescriptor*>::const_iterator it, end = _layers.end();
   for (it = _layers.begin(); it != end; ++it) {
      if ((*it) == gld) {
	 break;
      }
   }
   if (it == end) {
      throw SyntaxErrorException(
	 "NodeSet passed invalid layer in getNodes() method!");
   }

   unsigned coordIndex;

   // main loop
   VolumeOdometer vo(_beginCoords, _increment, _endCoords);
   std::vector<int> & coords = vo.look();
   for (; !vo.isRolledOver(); vo.next()) {
     coordIndex = _grid->getNodeIndex(coords);
     getNodesWithCoordinates(nodes, gld, coordIndex);
   }
}

const std::vector<GridLayerDescriptor*> & NodeSet::getLayers()
{
   // The original grid might have new layers, so we have to check.
   resetAllLayers();
   return _layers;
}

void NodeSet::setAllLayers()
{
   _allLayers = true;
   resetAllLayers();
}

void NodeSet::setLayers(const std::vector<GridLayerDescriptor*>& layers)
{
   std::vector<GridLayerDescriptor*>::const_iterator it, end = layers.end();
   for (it = layers.begin(); it != end; ++it) {
      if ((*it)->getGrid() != _grid) {
	 throw SyntaxErrorException(
	    "Layer added to NodeSet is not from specified NodeSet Grid!");
      }
   }
   _layers = layers;
   _allLayers = false;
}

void NodeSet::setLayers(const std::list<GridLayerDescriptor*>& layers)
{
   _layers.clear();
   std::list<GridLayerDescriptor*>::const_iterator it, end = layers.end();
   for (it = layers.begin(); it != end; ++it) {
      GridLayerDescriptor* gld = (*it);
      if (gld->getGrid() != _grid) {
	 throw SyntaxErrorException(
	    "Layer added to NodeSet is not from specified NodeSet Grid!");
      }
      _layers.push_back(gld);
   }
   _allLayers = false;
}

void NodeSet::addLayer(GridLayerDescriptor* gld)
{
   if (gld->getGrid() != _grid) {
      throw SyntaxErrorException(
	 "Layer added to NodeSet is not from specified NodeSet Grid!");
   }
   if (_allLayers == true) {
      _layers.clear();
      _allLayers = false;
   }
   _layers.push_back(gld);
}

void NodeSet::setIndices(const std::vector<int>& indices)
{
   if (indices.empty()) {
	 throw SyntaxErrorException(
	    "Tried to set indices on NodeSet with an empty array.");
   }

   _indices = indices;           // error checking here? JK
   sort(_indices.begin(), _indices.end());
   _allIndices = false;
}

void NodeSet::duplicate(std::auto_ptr<NodeSet>& dup) const
{
   dup.reset(new NodeSet(*this));
}

void NodeSet::resetAllLayers()
{
   if (_allLayers) {
      if (_layers.size() != _grid->getLayers().size()) {
	 _layers = _grid->getLayers();
      }
   }
}

void NodeSet::getNodesWithCoordinates(
   std::vector<NodeDescriptor*>& nodes, GridLayerDescriptor* gld, 
   unsigned coordIndex) const
{
   NodeAccessor* na = gld->getNodeAccessor();
   int maxDensityIndex = gld->getDensity(coordIndex);
   unsigned int j = 0;
   for(int i = 0; i < maxDensityIndex; ++i) {
      if (_allIndices) {
	 nodes.push_back(na->getNodeDescriptor(coordIndex, i));
      } else {
	 if (i == _indices[j]) {
	    if (gld->getDensity(coordIndex)>i) {
	       nodes.push_back(na->getNodeDescriptor(coordIndex,i));
	    }
	    j++;
	 }
      }
      if (!_allIndices && j >= _indices.size()) {
	 break;
      }
   }
}

bool NodeSet::haveCommonNode(const NodeSet& rv) const
{
   // Grids can fail
   if (_grid != rv._grid) {
      return false;
   }

   // indices can fail
   if (!_allIndices && !rv._allIndices) {
      // One matching index is good enough..
      bool found = false;
      std::vector<int>::const_iterator it, it2, end = _indices.end(), 
	 end2 = rv._indices.end();
      for (it = _indices.begin(); it != end; ++it) {
	 for (it2 = rv._indices.begin(); it2 != end2; ++it2) {
	    if (*it == *it2) {
	       found = true;
	       break;
	    }
	 }
	 if (found) break;
      }
      if (!found) return false;
   }      
   

   // boundaries can fail, begin and end are inclusive
   if (!isAllCoords() && !rv.isAllCoords()) {
      int begin1, end1, begin2, end2;
      unsigned size = _beginCoords.size();
      for (unsigned i = 0; i < size; ++i) {
	 begin1 = _beginCoords[i];
	 end1 = _endCoords[i];
	 begin2 = rv._beginCoords[i];
	 end2 = rv._endCoords[i];
	 // Check boundaries of one against each other for overlapping region
	 if ( !( ((begin2 >= begin1) && (begin2 <= end1)) || 
		 ((end2 >= begin1) && (end2 <= end1)) ) ) {
	    return false;
	 }
      }
   }


   // At this point nothing can fail, if we find a matching pair, 
   std::vector<GridLayerDescriptor*>::const_iterator it, it2, 
      end = _layers.end(), end2 = rv._layers.end();
   
   for(it = _layers.begin(); it != end; ++it) {
      for (it2 = rv._layers.begin(); it2 != end2; ++it2) {
	 if (*it == *it2) {
	    return true;
	    break;
	 }
      }
   }

   return false;
}
