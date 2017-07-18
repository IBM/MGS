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

//  A random order odometer returns coordinates from a hypercube randomly without replacement.
//  Use next() to draw a random coordinate.
#include "Copyright.h"
//  Once constructed, use look() to get the current coordinate without incrementing.
//  Use next() to increment FIRST then get the current coordinate.  So ...
//  Do not use next() to get the begin coordinate, unlesss you first reset()
//  Use isAtEnd() to check if odometer is at last coordinate
//	 Use isRolledOver() to check if odometer is at first coordinat

#ifndef RANDOMORDERODOMETER_H
#define RANDOMORDERODOMETER_H

#include "Odometer.h"
#include "rndm.h"
#include <vector>
#include <list>


class RandomOrderOdometer : public Odometer
{

   public:

  RandomOrderOdometer(std::vector<int> & begin, std::vector<int> & end, RNG&);
      bool isAtEnd();
      bool isRolledOver();
      std::vector<int> & look();
      std::vector<int> & next(RNG&);
      void reset(RNG&);
      int getSize();
      ~RandomOrderOdometer();

   private:

      std::vector<int> _begin;
      std::vector<std::vector<int> > _sample;
      std::vector<int> _end;
      std::vector<int> _current;
      bool _atEnd;
      bool _rolledOver;
      int _size, _abs_size;
};
#endif
