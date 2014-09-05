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

//  A vectored odometer returns coordinates from a hypercube in an order specified by a vector which is
//  of equal dimensionality as the hypercube, and which represents the step in each dimension the odometer
#include "Copyright.h"
//  takes as it traverses the hypercube. The vector is passed in during construction.
//  Once constructed, use look() to get the first or current coordinate without incrementing.
//  Use next() to increment FIRST then get the current coordinate.  So ...
//  Do not use next() to get the begin coordinate.
//  Use isAtEnd() to check if odometer is at any edge of hypercube, such that the next incremental step will roll it over.
//  Use isRolledOver() to check if odometer has moved torroidally from one of the hypercube edges to the opposite edge


// Example use in loop:

//    VectoredOdometer odmtr(beginCoords, endCoords);
//    for (coords = odmtr.look(); !odmtr.isRolledOver(); coords = odmtr.next() )
//    {
//       // print coords
//       cout <<coords;
//    }


#ifndef VECTOREDODOMETER_H
#define VECTOREDODOMETER_H

#include "Odometer.h"

#include <vector>


class VectoredOdometer : public Odometer
{

   public:

      VectoredOdometer(std::vector<int> & begin, std::vector<int> & end, std::vector<int> & steps);
      bool isAtEnd();
      bool isRolledOver();
      std::vector<int> & look();
      
      // Specific to VectoredOdometers only
      void set(std::vector<int> & coords);
      std::vector<int> & next();
      void reset();
      int getSize();
      ~VectoredOdometer();

   private:

      std::vector<int> _begin;
      std::vector<int> _current;
      std::vector<int> _end;
      std::vector<int> _steps;
      std::vector<int> _dims;
      bool _atEdge;
      bool _rolledOver;
      int _size;
};
#endif
