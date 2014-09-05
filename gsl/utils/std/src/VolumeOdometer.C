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

#include "VolumeOdometer.h"
#include "VectorOstream.h"
#include <cassert>

VolumeOdometer::VolumeOdometer(const std::vector<int> & begin, 
			       const std::vector<int> & end)
   : _current(begin), _begin(begin), _end(end)
{
   assert(begin.size() == end.size());
   unsigned sz = _begin.size();
   _increment.resize(sz, 1);

   _used = false;
   if (sz == 0) _size = 0;
   else {
      _size = 1;
      for (unsigned i = 0; i<sz; ++i) {
         _size *= (_end[i]-_begin[i]+1);
      }
   }

}

VolumeOdometer::VolumeOdometer(const std::vector<int> & begin, 
			       const std::vector<int> & increment,
			       const std::vector<int> & end)
   : _current(begin), _begin(begin), _increment(increment), _end(end)
{
   assert(begin.size() == end.size());

   _used = false;
   if (_begin.size() == 0) _size = 0;
   else {
      _size = 1;
      unsigned end = _begin.size();
      for (unsigned i = 0; i<end; ++i) {
         _size *= (_end[i]-_begin[i]+1);
      }
   }

}

bool VolumeOdometer::isAtEnd()
{
   return (_current == _end);
}

bool VolumeOdometer::isRolledOver()
{
   return (_used && (_current == _begin) );
}

std::vector<int> & VolumeOdometer::look()
{
   return _current;
}

std::vector<int> & VolumeOdometer::next()
{
   int wheel = _current.size()-1;
   int rollover;
   do {
      rollover = 0;
      _current[wheel]+=_increment[wheel];
      if (_current[wheel] > _end[wheel]) {
         _current[wheel] = _begin[wheel];
         if (--wheel >= 0) {     // wheel is < 0 only when incrementing odometer from end to begin
            rollover = 1;
         }
      }
   } while (rollover);
   _used = true;
   return _current;
}

void VolumeOdometer::reset()
{
   _current = _begin;
   _used = false;
}

int VolumeOdometer::getSize()
{
   return _size;
}

VolumeOdometer::~VolumeOdometer()
{
}
