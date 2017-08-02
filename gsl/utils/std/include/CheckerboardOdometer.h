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
//  starting at the begin coordinate of the odometer itself.  The stepsize of each check
//  is also specified during construction, as well as whether or not the odometer will return
//  incomplete sections at the edges. Both the beginning and end coordinates of the sub-hypercubes
//  remain accessible by reference throughout one begin/end cycle of next()

//  Once constructed, use look() to get the first or current coordinate without incrementing.
//  Use next() to increment FIRST then get the current coordinate.  So ...
//  Do not use next() to get the begin coordinate.
//  Use isAtEnd() to check if odometer is at last coordinate
//  Use isRolledOver() to check if odometer is at first coordinat
/*
Example use in loop:

   CheckerboardOdometer odmtr(CheckerboardOdometer::_NO_PARTIAL_EDGES, beginCoords, endCoords, sectionSize, stepSize);
   for (coords = odmtr.look(); !odmtr.isRolledOver(); coords = odmtr.next() )
   {
      // print coords
      cout <<coords;
   }

*/

#ifndef CHECKERBOARDODOMETER_H
#define CHECKERBOARDODOMETER_H

#include "Odometer.h"

#include <vector>


class CheckerboardOdometer : public Odometer
{

   public:
      enum Type {_PARTIAL_EDGES, _NO_PARTIAL_EDGES};

      CheckerboardOdometer(Type t, std::vector<int> & begin, std::vector<int> & end, std::vector<int> & sectionSize, std::vector<int> & stepSize);
      virtual bool isAtEnd();
      virtual bool isRolledOver();
      virtual std::vector<int> & look();
      virtual std::vector<int> & next();
      virtual void reset();
      virtual int getSize();
      virtual ~CheckerboardOdometer();

   protected:

      Type _type;
      std::vector<int> _begin;
      std::vector<int>* _current;
      std::vector<int>* _currentSectionBegin;
      std::vector<int> _end;
      std::vector<int> _sectionSize;
      std::vector<int> _stepSize;
      std::vector<int> _startPos;
      std::vector<int> _offsets;
      std::vector<int> _borders;
      bool _isCurrentSectionBegin;
      bool _used;
      int _size;
};
#endif
