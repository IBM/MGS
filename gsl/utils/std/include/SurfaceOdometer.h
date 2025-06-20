// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
//  A surface odometer returns coordinates from the surface of a hypercube.  Sides of the
//  hypercube are explored in pairs, bottom then top.
#include "Copyright.h"
//  Once constructed, use look() to get the first or current coordinate without incrementing.
//  Use next() to increment FIRST then get the current coordinate.  So ...
//  Do not use next() to get the begin coordinate.
//  Use isAtEnd() to check if odometer is at last coordinate
//	 Use isRolledOver() to check if odometer is at first coordinat

#ifndef SURFACEODOMETER_H
#define SURFACEODOMETER_H

#include "Odometer.h"

#include <vector>


class SurfaceOdometer : public Odometer
{

   public:

      SurfaceOdometer(std::vector<int> & begin, std::vector<int> & end);
      bool isAtEnd();
      bool isRolledOver();
      std::vector<int> & look();
      std::vector<int> & next();
      void reset();
      int getSize();
      ~SurfaceOdometer();

   private:

      std::vector<int> _begin;
      std::vector<int> _current;
      std::vector<int> _end;
      std::vector<int> _abs_begin;
      std::vector<int> _abs_end;
      std::vector<int> _atEnd;
      int _fixedWheel;           // specifies the dimension that is collapsed to sample the sides along this axis
      int _val;                  // stores the begin or end value for the dimension that is collapsed
      bool _side;                // each dimension has two sides "top" and "bottom" here defined as true and false'
      bool _used;
      int _size;
};
#endif
