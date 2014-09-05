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

#ifndef GRIDSET_H
#define GRIDSET_H
#include "Copyright.h"

#include <vector>
#include <iostream>

class Grid;
class Node;
class GridSet;

std::ostream & operator<<(std::ostream&,GridSet const&);

class GridSet
{
   public:
      // constructors/destructor
      GridSet(Grid* grid);
      virtual ~GridSet();

      // coordinate methods
      // Note: all coords are inclusive!
      // That is, both begin and end coordinates are valid
      // and the size of each dimension is end[i]-begin[i] + 1

      bool isAllCoords() const {
	 return _allCoords;
      }

      void setAllCoords();

      void setCoords(const std::vector<int>& beginCoords,
		     const std::vector<int>& endCoords);

      void setCoords(const std::vector<int>& beginCoords,
		     const std::vector<int>& increment,
		     const std::vector<int>& endCoords);

      const std::vector<int>& getBeginCoords() const {
	 return _beginCoords;
      }

      const std::vector<int>& getIncrement() const {
	 return _increment;
      }

      const std::vector<int>& getEndCoords() const {
	 return _endCoords;
      }

      // utility methods

      // returns a vector of indices from Grid::getNodeIndex() method calls
      // derived from the beginCoords and endCoords
      void getGridNodeIndices(std::vector<int>& gridNodeIndices);
      virtual std::ostream& print(std::ostream &os, GridSet const &gs) const;

      // Grid methods
      Grid* getGrid() {
	 return _grid;
      }
      
   protected:
      bool _allCoords;
      std::vector<int> _beginCoords;
      std::vector<int> _increment;
      std::vector<int> _endCoords;
      Grid* _grid;
};
#endif
