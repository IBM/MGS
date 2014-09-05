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

#include "RandomOrderOdometer.h"
#include "VolumeOdometer.h"
#include "VectorOstream.h"
#include "rndm.h"
#include <stdlib.h>

RandomOrderOdometer::RandomOrderOdometer(std::vector<int> & begin, std::vector<int> & end)
: _begin(begin), _end(end)
{
   _atEnd = false;
   _rolledOver = false;
   VolumeOdometer volOdmtr(_begin, _end);
   std::vector<int> & vec = volOdmtr.look();
   for (; !volOdmtr.isRolledOver(); volOdmtr.next() )
      _sample.push_back(vec);
   _size = _abs_size = _sample.size();
   next();                       // Must call next() to assign current its initial value, and allow for initial look()
}


bool RandomOrderOdometer::isAtEnd()
{
   return _atEnd;
}


bool RandomOrderOdometer::isRolledOver()
{
   return _rolledOver;
}


std::vector<int> & RandomOrderOdometer::look()
{
   return _current;
}


std::vector<int> & RandomOrderOdometer::next()
{
   _rolledOver = false;
   if (_atEnd) {                 // last draw was the end
      reset();                   // so reset the size counter using reset()
      _rolledOver = true;        // and note that this next() is the rolled over value
   }
   // draw is int from 0.._size-1
//   int draw = Rangen.irandom32(0, _size-1);
   int draw = irandom(0, _size-1);
   _current = _sample[draw];
   if (--_size > 0) {            // if decremented size is now zero, the last draw was from a sample of 1
      // otherwise transfer the end to the position of the draw
      _sample[draw] = _sample[_size];
      // and the draw (current) to the end
      _sample[_size] = _current;
   }
   else _atEnd = true;           // last draw was from a sample of 1 so that's the end
   return _current;
}


void RandomOrderOdometer::reset()
{
   _atEnd = false;
   _rolledOver = false;
   _size = _abs_size;
   next();                       // Must call next() to assign current its initial value, and allow for initial look()
}


int RandomOrderOdometer::getSize()
{
   return _abs_size;
}


RandomOrderOdometer::~RandomOrderOdometer()
{
}
