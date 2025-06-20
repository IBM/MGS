// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "NodeType.h"
//#include "Node.h"
#include "NodeAccessor.h"
//#include "NDPair.h"
#include "VectorOstream.h"

GridLayerDescriptor::GridLayerDescriptor(
   Grid* grid, const std::vector<int>& densityVector, 
   std::string name, NodeType* nt,  NDPairList const & ndpList, unsigned granuleMapperIndex)
   : _grid(grid), _nodeAccessor(0), _name(name), _nt(nt),
     _densityVector(densityVector), _uniformDensity(0), _ndpList(ndpList),
     _granuleMapperIndex(granuleMapperIndex)
{   
   if (_densityVector.size() == 1) {
      _uniformDensity = _densityVector[0];
   }
   std::unique_ptr<NodeAccessor> nodeAccessor;
   _nt->getNodeAccessor(std::move(nodeAccessor), this);
   setNodeAccessor(nodeAccessor);
}

// Return the string representation of the model name
// e.g. in GSL we declare:
// NodeType HodgkinHuxleyVoltage(...){...}
// then 'HodgkinHuxleyVoltage' is the model name
std::string GridLayerDescriptor::getModelName()
{
   return _nt->getModelName();
}

// Return the #of instances to be created at 
// the given grid's index position 'nodeIndex'
// for this GridLayer
int GridLayerDescriptor::getDensity(int nodeIndex)
{
   int retval = _uniformDensity;
   if (!_uniformDensity) {
      retval= _densityVector[nodeIndex % _densityVector.size()];
   }  
   // This implementation allows densityVector to "wrap" the grid; 
   // if density Vector MUST have size == 1 or size == grid points, 
   // error checking should occur here instead
   return retval;
}

int GridLayerDescriptor::getDensity(std::vector<int> const & coords)
{
   return getDensity(_grid->getNodeIndex(coords));
}

unsigned GridLayerDescriptor::getMaxDensity()
{
   int rval = _uniformDensity;
   if (!_uniformDensity) {
      std::vector<int>::iterator it, end = _densityVector.end();
      for (it = _densityVector.begin(); it != end; ++it) {
	 if ((*it)>rval) rval = (*it);
      }
   }
   return rval;
}

unsigned GridLayerDescriptor::getMinDensity()
{
   int rval = getMaxDensity();
   std::vector<int>::iterator it, end = _densityVector.end();
   for (it = _densityVector.begin(); it != end; ++it) {
      if ((*it)<rval) rval = (*it);
   }
   return rval;
}

void GridLayerDescriptor::replaceDensityVector(unsigned* replacement, int size)
{
  assert(size>0);
  if (size==1) {
    _uniformDensity=replacement[0];
    _densityVector.push_back(_uniformDensity);
  }
  else {
    assert(size==_densityVector.size());
    for (int i=0; i<size; ++i) {
      _uniformDensity=0;
      _densityVector[i]=replacement[i];
    }
  }
}

void GridLayerDescriptor::setNodeAccessor(std::unique_ptr<NodeAccessor>& na)
{
   _nodeAccessor = na.release();
}

GridLayerDescriptor::~GridLayerDescriptor()
{
   delete _nodeAccessor;
}
