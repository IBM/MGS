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

#ifndef GRID_H
#define GRID_H
#include "Copyright.h"

#include <string>
#include <vector>
#include <memory>
#include "ShallowArray.h"

class GridLayerDescriptor;
class Repertoire;
class NodeType;
class NDPairList;

// SGC comments [begin]
// A grid is a collection of layers (GridLayerDescriptors. 
// The dimensions of the grid is set at construction, however each layer
// may have its own density. 
// What it looks like, the main responsibility of the grid is to figure
// out how to map node coordinates to one dimensional coordinates(nodeIndex)
// and vice versa.
// It also keeps track of the maximum and the minimum densities out of 
// all the layers it has.
// SGC comments [end]


class Grid : public std::vector<GridLayerDescriptor*>
{

   public:
      Grid(const std::vector<int>& size);
      virtual ~Grid();

      std::string getName();
      void getSize2Dim(int& x, int& y) const;
      GridLayerDescriptor* getLayer(std::string layerName) const;

      GridLayerDescriptor* addLayer(const std::vector<int>& densityVector, 
				    std::string name, NodeType* nt, 
				    NDPairList& ndpList, int granuleMapperIndex);

      unsigned getNodeIndex(const std::vector<int>& coords) const;
      unsigned getNodeIndex(const ShallowArray<unsigned>& coords) const;
      void getNodeCoords(int nodeIndex, std::vector<int>& coords) const;
      void getNodeCoords(int nodeIndex, 
			 ShallowArray<unsigned, 3, 2>& coords) const;
      void getNodeCoords(int nodeIndex, int& x, int& y) const;

      unsigned getDimensions() const {
	 return _dimensions;
      }

      unsigned getNbrGridNodes() const {
	 return _gridNodes;
      }

      unsigned getNumLayers() const {
	 return size();
      }

      unsigned getMaxDensity() const {
	 return _maxDensity;
      }

      unsigned getMinDensity() const {
	 return _minDensity;
      }

      Repertoire* getParentRepertoire() const {
	 return _parentRepertoire;
      }

      void setParentRepertoire(Repertoire* parentRepertoire) {
	 _parentRepertoire = parentRepertoire;
      }
      
      const std::vector<int>& getSize() const {
	 return _size;
      }

      const std::vector<GridLayerDescriptor*>& getLayers() const {
	 return *this;
      }

      // will throw a SyntaxErrorException if the coordinates 
      // can't belong to the grid.
      void checkCoordinateSanity(const std::vector<int>& coords) const;

   private:
      void addLayer(std::auto_ptr<GridLayerDescriptor>&);

      unsigned _dimensions;
      unsigned _gridNodes;
      unsigned *_strides;
      Repertoire* _parentRepertoire;
      std::vector<int> _size;
      unsigned _maxDensity;
      unsigned _minDensity;
};

#endif
