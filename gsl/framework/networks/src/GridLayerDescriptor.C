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
   std::auto_ptr<NodeAccessor> nodeAccessor;
   _nt->getNodeAccessor(nodeAccessor, this);
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

void GridLayerDescriptor::setNodeAccessor(std::auto_ptr<NodeAccessor>& na)
{
   _nodeAccessor = na.release();
}

GridLayerDescriptor::~GridLayerDescriptor()
{
   delete _nodeAccessor;
}
