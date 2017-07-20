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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "GridSet.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "VolumeOdometer.h"
#include "Repertoire.h"
#include "SyntaxErrorException.h"

#include <sstream>
#include <iostream>
std::ostream & operator<<(std::ostream& os,GridSet const& gs)
{
   return gs.print(os,gs);
}

std::ostream & GridSet::print(std::ostream & os, GridSet const & gs) const
{
   // print path
   Repertoire* current;
   Repertoire* fromRep = _grid->getParentRepertoire();

   std::list<Repertoire*> path;

   // build Repertoire path from leaf to root for "from" Node
   for(current = fromRep; current!=0; 
       current = current->getParentRepertoire()) {
      path.push_back(current);
   }

   // start at root and work toward leaf
   std::list<Repertoire*>::reverse_iterator fi, 
      fBegin = path.rbegin(), fEnd = path.rend();
   for (fi = fBegin; fi != fEnd; ++fi) {
      if (fi != fBegin) {
	 os<<"/";
      }
      os << (*fi)->getName();
   }

   // print bounds
   char token = '[';
   for(unsigned int i = 0; i < _beginCoords.size(); ++i) {
     os << token << _beginCoords[i] << ":" << _increment[i] << ":" << _endCoords[i];
      if (token == '[')  {
	 token = ',';
      }
   }
   os << ']';
   return os;
}


void GridSet::setAllCoords()
{
   _beginCoords.clear();
   _increment.clear();
   _endCoords.clear();
   const std::vector<int>& end = _grid->getSize();
   int coordsSize = end.size();
   for (int i = 0; i < coordsSize; ++i) {
      _beginCoords.push_back(0);
      _increment.push_back(1);
      _endCoords.push_back(end[i]-1);
   }
}


GridSet::GridSet(Grid * grid)
   : _allCoords(true), _grid(grid)
{
   const std::vector<int>& end = _grid->getSize();
   int coordsSize = end.size();
   for (int i = 0; i < coordsSize; i++) {
      _beginCoords.push_back(0);
      _increment.push_back(1);
      _endCoords.push_back(end[i]-1);
   }
}

void GridSet::getGridNodeIndices(std::vector<int>& gridNodeIndices)
{
   gridNodeIndices.clear();
   VolumeOdometer vo(_beginCoords, _increment, _endCoords);
   std::vector<int>& coords = vo.look();
   for (; !vo.isRolledOver(); vo.next()) {
      gridNodeIndices.push_back(_grid->getNodeIndex(coords));
   }
}

GridSet::~GridSet()
{
}

void GridSet::setCoords(const std::vector<int>& beginCoords, 
			const std::vector<int>& endCoords)
{

   _allCoords = false;
   _grid->checkCoordinateSanity(beginCoords);
   _grid->checkCoordinateSanity(endCoords);
   // They have the same size if the top 2 tests are passed.
   std::vector<int>::const_iterator it, it2, end = beginCoords.end();
   for (it = beginCoords.begin(), it2 = endCoords.begin(); it != end; 
	++it, ++it2) {
      if (*it2 < *it) {
	 std::ostringstream os; 
	 os << "The end coordinate " << *it2 
	    << " is smaller then the begin coordinate " << *it;
	 throw SyntaxErrorException(os.str());
      }
      _increment.push_back(1);
   }
   _beginCoords = beginCoords;
   _endCoords = endCoords;
}

void GridSet::setCoords(const std::vector<int>& beginCoords, 
			const std::vector<int>& increment,
			const std::vector<int>& endCoords)
{

   _allCoords = false;
   _grid->checkCoordinateSanity(beginCoords);
   _grid->checkCoordinateSanity(endCoords);
   // They have the same size if the top 2 tests are passed.
   std::vector<int>::const_iterator it, it2, end = beginCoords.end();
   for (it = beginCoords.begin(), it2 = endCoords.begin(); it != end; 
	++it, ++it2) {
      if (*it2 < *it) {
	 std::ostringstream os; 
	 os << "The end coordinate " << *it2 
	    << " is smaller then the begin coordinate " << *it;
	 throw SyntaxErrorException(os.str());
      }
   }
   _beginCoords = beginCoords;
   _increment = increment;
   _endCoords = endCoords;
}
