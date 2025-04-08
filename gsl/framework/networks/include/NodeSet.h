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

#ifndef NODESET_H
#define NODESET_H
#include "Copyright.h"

#include "GridSet.h"

#include <vector>
#include <list>
#include <string>
#include <memory>


class GridLayerDescriptor;
class Grid;

class NodeDescriptor;

class NodeSet : public GridSet
{
   public:
      // constructors/destructor
      NodeSet(Grid* grid);
      NodeSet(const GridSet& gridSet);
      NodeSet(NodeDescriptor* n);
      NodeSet(Grid* grid, std::vector<NodeDescriptor*> nodes);
      ~NodeSet();

      // utility
      bool contains(NodeDescriptor* n);

      // Node methods
      void getNodes(std::vector<NodeDescriptor*>& nodes);
      void getNodes(std::vector<NodeDescriptor*>& nodes, std::string gldName);
      void getNodes(std::vector<NodeDescriptor*>& nodes, 
		    GridLayerDescriptor* gld);

      // layer methods
      bool isAllLayers() {
	 return _allLayers;
      }

      bool haveCommonNode(const NodeSet& rv) const;

      void setAllLayers();
      void addLayer(GridLayerDescriptor*);
      void setLayers(const std::vector<GridLayerDescriptor*>& layers);
      void setLayers(const std::list<GridLayerDescriptor*>& layers);
      const std::vector<GridLayerDescriptor*>& getLayers();

      // indices methods

      // check this before getIndices
      bool isAllIndices() {
	 return _allIndices;
      }

      // set indices to include if they actually exist
      void setIndices(const std::vector<int>& indices);

      void setAllIndices() {
	 _allIndices = true;
      }

      // only use if isAllIndices false, 
      // be sure to check for existence of nodes
      const std::vector<int>& getIndices() {
	 return _indices;
      }

      void empty() {
	setLayers(std::vector<GridLayerDescriptor*>());
      }
      
      virtual void duplicate(std::unique_ptr<NodeSet>&& dup) const;

   protected:
      void resetAllLayers();
      inline void getNodesWithCoordinates(
	 std::vector<NodeDescriptor*>& nodes, GridLayerDescriptor* gld,
	 unsigned coordIndex) const;

      bool _allLayers;
      bool _allIndices;
      std::vector<int> _indices;
      std::vector<GridLayerDescriptor*> _layers;
#ifdef TEST_MULTIPLE_THREADS
      int _numPartitions;
   public:
      void setNumPartitions(int partitionCount);
      void getNodes(std::vector<NodeDescriptor*>& nodes, int partitionIdx);
#endif
};
#endif
