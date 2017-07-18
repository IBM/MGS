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

//  A checkerboard odometer returns coordinates from a hypercube that demarcate the begining
//  and ends of prespecified sub-hypercubes of the odometer's hypercube.  The prespecification
#include "Copyright.h"
//  is accomplished through the third and fourth vectors passed into the constructor which must be
//  of equal dimensionality as the odometer itself.  The section size specifies a
//  a hypercube size, whose begin and end coords will be returned in order using next(),
//  starting at the begin coordinate of the odometer itself.  The stepsize across the torroidal checkerboard
//  is also specified during construction, as well as whether or not the odometer will return
//  incomplete sections at the edges. Both the beginning and end coordinates of the sub-hypercubes
//  remain accessible by reference throughout one begin/end cycle of next().  Finally, the start position
//  is specified for where the checkerboard odometer begins (coordinate is for lower left corner of first check).

//  Once constructed, use look() to get the first or current coordinate without incrementing.
//  Use next() to increment FIRST then get the current coordinate.  So ...
//  Do not use next() to get the begin coordinate.
//  Use isAtEnd() to check if odometer is at last coordinate
//  Use isRolledOver() to check if odometer is at first coordinat

#ifndef VECTOREDCHECKERBOARDODOMETER_H
#define VECTOREDCHECKERBOARDODOMETER_H
#include "CheckerboardOdometer.h"

#include <vector>


class VectoredCheckerboardOdometer : public CheckerboardOdometer
{

   public:
      VectoredCheckerboardOdometer(CheckerboardOdometer::Type t, std::vector<int> & begin, std::vector<int> & end, std::vector<int> & sectionSize,
         std::vector<int> & stepSize, std::vector<int> & startPos);
      bool isAtEnd();
      bool isRolledOver();
      std::vector<int> & look();
      std::vector<int> & next();
      void reset();
      int getSize();
      ~VectoredCheckerboardOdometer();
   private:
      std::vector<int> _startPos;
};
#endif
