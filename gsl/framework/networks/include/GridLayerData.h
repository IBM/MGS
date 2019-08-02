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

#ifndef GridLayerData_H
#define GridLayerData_H
#include "Copyright.h"

#include <vector>
#include <cstddef>

class Node;
class NodeDescriptor;
class Edge;
class GridLayerDescriptor;
class NodeCompCategoryBase;
class NodeRelationalDataUnit;

// SGC comments [begin]
// This is the class that owns the nodes, each node type has its own 
// CG_XGridLayerData that inherits from this class. That inherited class
// will own the nodes, i.e., will have a variable X* nodes;
// The inerited class's constructor will allocate the right amount of nodes
// and destructor will delete them.
// Having a Node* at this class just won't work because it is a regular C
// array, It could be possible though, if an stl vector or something of that
// nature was used.
// SGC comments [end]

class GridLayerData
{
   public:
      GridLayerData(NodeCompCategoryBase* compCategory, 
		    GridLayerDescriptor* gridLayerDescriptor,
		    int gridLayerIndex);
      virtual ~GridLayerData();

      size_t getNbrUnits() 
      {
	 return _nbrUnits;
      }

      size_t getNbrNodesAllocated() 
      {
	 return _nbrNodesAllocated;
      }
      void incrementNbrNodesAllocated() {
	++_nbrNodesAllocated;
      }
      void setNbrNodesAllocated(size_t size) {
	_nbrNodesAllocated = size;
      }

      GridLayerDescriptor* getGridLayerDescriptor() {
	 return _gridLayerDescriptor;
      }

/*
      CompCategory* getCompCategory() {
	 return _compCategory;
      }
*/

      NodeCompCategoryBase* getNodeCompCategoryBase() {
	 return _compCategory;
      }

      const std::vector<int>& getNodeOffsets() {
	 return _nodeOffsets;
      }

      int getGridLayerIndex() const {
	 return _gridLayerIndex;
      }

   protected:
      //int _nbrUnits;
      size_t _nbrUnits;
      //int _nbrNodesAllocated; // _nbrNodeAllocated == _nbrUnits in non-distributed environment; <= _nbrUnits otherwise 
      size_t _nbrNodesAllocated; // _nbrNodeAllocated == _nbrUnits in non-distributed environment; <= _nbrUnits otherwise 
      GridLayerDescriptor *_gridLayerDescriptor;
      NodeCompCategoryBase* _compCategory;
      std::vector<int> _nodeOffsets;     
      int _gridLayerIndex;

   private:
      // Disable, because we can't duplicate an array, to make this work,
      // we either should use something like a deque or a vector instead of
      // a C array for nodes, or make every node serve an C array of its 
      // size.
      GridLayerData(const GridLayerData& rv) {};
      // Disable, same as above.
      GridLayerData& operator=(const GridLayerData& rv) { return *this;};

};

#endif
