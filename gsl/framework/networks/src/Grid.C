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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "Grid.h"
#include "GridLayerDescriptor.h"
//#include "Node.h"
#include "Repertoire.h"
//#include "NDPairList.h"
#include "SyntaxErrorException.h"

#include <limits.h>
#include <sstream>
#include <iostream>

Grid::Grid(const std::vector<int>& size) 
   : _dimensions(size.size()), _size(size), _maxDensity(0), 
     _minDensity(UINT_MAX)
{
   // set up strides vector for fast computation of gridnode 
   // coordinates and indexes
   if (_dimensions < 1) {
      std::string mes = "Attempt to create a Grid with no dimensions!";
      std::cerr << mes << std::endl;
      throw SyntaxErrorException(mes);
   }
   
   for (unsigned i = 0; i < _dimensions; ++i) {
      if (size[i] <= 0) {
	 std::ostringstream os;
	 os << "The size of dimension " << i << " is " << size[i] 
	    << " it should have been greater than 0.";
	 throw SyntaxErrorException(os.str());
      }
   }

   _strides = new unsigned[_dimensions];
   _gridNodes = 1;
   // be careful i has to be int, due to --i
   for (int i = _dimensions - 1; i >= 0; --i) {
      _strides[i] = _gridNodes;
      _gridNodes *= _size[i];
   }
}

std::string Grid::getName()
{
   return _parentRepertoire->getName() + "Grid";
}

void Grid::getSize2Dim(int& x, int& y) const
{
   if (_dimensions != 2) {
      std::string mes = "The grid should be 2 dimensional for Grid::";
      mes += "getSize2Dim(int& x, int& y) to be used.";

      std::cerr << mes << std::endl;
      throw SyntaxErrorException(mes);
   }
   x = _size[0];
   y = _size[1];
}

GridLayerDescriptor* Grid::getLayer(std::string layerName) const
{
   GridLayerDescriptor* rval = 0;
   Grid::const_iterator it, end = this->end();
   for (it = this->begin(); it != end; ++it) {
      if ((*it)->getName() == layerName) {
         rval = (*it);
         break;
      }
   }
   if (rval == 0) {
      std::string mes = "Specified layer ";
      mes += layerName;
      mes += " not found in Grid!";
      std::cerr <<  mes << std::endl;
      throw SyntaxErrorException(mes);
   }
   return rval;
}

GridLayerDescriptor* Grid::addLayer(const std::vector<int>& densityVector, 
				    std::string name, NodeType* nt, 
				    NDPairList & ndpList,
				    int granuleMapperIndex)
{
   GridLayerDescriptor *gld = new GridLayerDescriptor(
      this, densityVector, name, nt, ndpList, granuleMapperIndex);
   std::unique_ptr<GridLayerDescriptor> gld_ap(gld);
   addLayer(gld_ap);
   return gld;
}

void Grid::addLayer(std::unique_ptr<GridLayerDescriptor> & layer)
{
   if (layer->getMaxDensity() > _maxDensity) {
      _maxDensity = layer->getMaxDensity();
   }
   if (layer->getMinDensity() < _minDensity || this->size() == 0) {
      _minDensity = layer->getMinDensity();
   }
   this->push_back(layer.release());
}

unsigned Grid::getNodeIndex(const std::vector<int>& coords) const
{
   if (coords.size() != _dimensions) {
      std::ostringstream os;
      os << "Attempt to getNodeIndex with coordinates of the wrong size!: "
	 << coords.size() << " , " << _dimensions;
      std::cerr << os.str() << std::endl;
      throw SyntaxErrorException(os.str());      
   }
   std::vector<int>::const_iterator ci = coords.begin();
   unsigned *si = _strides;
   unsigned *end = &_strides[_dimensions];
   unsigned index = 0;
   do {
      index += *si++ * *ci++;
   } while(si < end);
   return index;
}

unsigned Grid::getNodeIndex(const ShallowArray<unsigned>& coords) const
{
   if (coords.size()!= _dimensions) {
      std::ostringstream os;
      os << "Attempt to getNodeIndex with coordinates of the wrong size!: "
	 << coords.size() << " , " << _dimensions;
      std::cerr << os.str() << std::endl;
      throw SyntaxErrorException(os.str());      
   }
   ShallowArray<unsigned>::const_iterator ci = coords.begin();
   unsigned *si = _strides;
   unsigned *end = &_strides[_dimensions];
   unsigned index = 0;
   do {
      index += *si++ * *ci++;
   } while(si < end);
   return index;
}

void Grid::getNodeCoords(int nodeIndex, std::vector<int>& coords) const
{
   coords.clear();
   unsigned d;
   unsigned *si = _strides;
   unsigned *end = &_strides[_dimensions];
   int remainder = nodeIndex;
   do {
      d = remainder/ *si;
      coords.push_back(d);
      remainder -= d * *si++;
   } while(si < end);
}

void Grid::getNodeCoords(int nodeIndex, 
			 ShallowArray<unsigned, 3, 2>& coords) const
{
   coords.clear();
   unsigned d;
   unsigned *si = _strides;
   unsigned *end = &_strides[_dimensions];
   int remainder = nodeIndex;
   do {
      d = remainder/ *si;
      coords.push_back(d);
      remainder -= d * *si++;
   } while(si < end);
}

void Grid::getNodeCoords(int nodeIndex, int& x, int& y) const
{
   if (_dimensions != 2) {
      std::ostringstream os;
      os << "The grid should be 2 dimensional for Grid::"
	 << "getSize2Dim(int& x, int& y) to be used.";
      std::cerr << os.str() << std::endl;
      throw SyntaxErrorException(os.str());      
   }

   unsigned *si = _strides;
   int remainder = nodeIndex;
   
   x = remainder/ *si;
   remainder -= x * *si++;
   y = remainder/ *si;
}

void Grid::checkCoordinateSanity(const std::vector<int>& coords) const
{
   if (coords.size() != _dimensions) {
      std::ostringstream os;
      os << "The requested coordinates have " << coords.size() 
	 << " dimensions while the grid has " << _dimensions << " dimensions.";
	 throw SyntaxErrorException(os.str());
   }
   for (unsigned i = 0; i < _dimensions; ++i) {
      if (coords[i] < 0) {
	 std::ostringstream os;
	 os << "The coordinate " << i << "'s value is " << coords[i] 
	    << ", it can not be a negative value.";
	 throw SyntaxErrorException(os.str());
      }
      if (coords[i] >= _size[i]) {
	 std::ostringstream os;
	 os << "The coordinate " << i << "'s value is " << coords[i] 
	    << ", it should have been smaller than " << _size[i] << ".";
	 throw SyntaxErrorException(os.str());
      }
   }
}

Grid::~Grid()
{
   delete [] _strides;
   Grid::iterator it, end = this->end();
   for (it = this->begin(); it != end; ++it) {
      delete *it;
   }
}
//
//inline unsigned Grid::getDimensions() const 
//{
//   return _dimensions;
//}
//inline unsigned Grid::getNbrGridNodes() const {
//   return _gridNodes;
//}
//inline unsigned Grid::getNumLayers() const {
//   return size();
//}
//inline unsigned Grid::getMaxDensity() const {
//   return _maxDensity;
//}
//inline unsigned Grid::getMinDensity() const {
//   return _minDensity;
//}
//inline Repertoire* Grid::getParentRepertoire() const {
//   return _parentRepertoire;
//}
//inline void Grid::setParentRepertoire(Repertoire* parentRepertoire) {
//   _parentRepertoire = parentRepertoire;
//}
//inline const std::vector<int>& Grid::getSize() const {
//   return _size;
//}
//inline const std::vector<GridLayerDescriptor*>& Grid::getLayers() const {
//   return *this;
//}
