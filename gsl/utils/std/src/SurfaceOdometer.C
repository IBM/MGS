// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SurfaceOdometer.h"
#include "VectorOstream.h"

SurfaceOdometer::SurfaceOdometer(std::vector<int> & begin, std::vector<int> & end)
: _begin(begin),_current(begin), _end(end), _abs_begin(begin), _abs_end(end), _atEnd(end)
{
   // collapse least significant dimension first
   _fixedWheel = _current.size()-1;
   _side = false;                // start at bottom (false)
   _used = false;                // not used yet
   _val = _end[_fixedWheel];     // must store what's being collapsed for retrieval later
   // collapse down to bottom
   _end[_fixedWheel] = _begin[_fixedWheel];
   for (unsigned i=1; i<_atEnd.size(); i++)
      _atEnd[i]--;               // create pattern to match to test if odometer is complete
   // set up size
   if (_abs_begin.size() == 0) _size = 0;
   else {
      _size = 1;
      for (unsigned i = 0; i<_abs_begin.size(); i++) {
         _size *= (_abs_end[i]-_abs_begin[i]+1);
      }
      int centerSize = 1;
      for (unsigned i = 0; i<_abs_begin.size(); i++) {
         /* +1-2 */
         centerSize *= (_abs_end[i]-_abs_begin[i]-1 );
      }
      if (centerSize>0) _size -= centerSize;
   }
}


bool SurfaceOdometer::isAtEnd()
{
   return (_current == _atEnd);
}


bool SurfaceOdometer::isRolledOver()
{
   return (_used && (_current == _abs_begin) );

}


std::vector<int> & SurfaceOdometer::look()
{
   return _current;
}


std::vector<int> & SurfaceOdometer::next()
{
   int wheel = _current.size()-1;
   int rollover;
   do {
      rollover = 0;
      _current[wheel]++;
      if (_current[wheel] > _end[wheel]) {
         _current[wheel] = _begin[wheel];
         // this line marks the end of portion identical with VolumeOdometer
         if (--wheel >= 0) rollover = 1;
         else {
            if (!_side) {        // still have the top to do along this axis
               // restore end from stored value
               _end[_fixedWheel] = _val;
               // all coords from _begin[_fixedWheel] now done, so increment and store it
               _val = _begin[_fixedWheel]+1;
               // collapse up to top
               _begin[_fixedWheel] = _end[_fixedWheel];
               _current = _begin;
               _side = true;     // move to top side
            }
            // top and bottom done
            else if (_fixedWheel > 0) {
               // restore begin from stored value
               _begin[_fixedWheel] = _val;
               // all coords from _end[_fixedWheel] now done, so decrement
               _end[_fixedWheel]--;
               _fixedWheel--;
               // store end from new dimension as is
               _val = _end[_fixedWheel];
               // collapse down to bottom
               _end[_fixedWheel] = _begin[_fixedWheel];
               _current = _begin;
               // move to bottom side
               _side = false;
            }
            else reset();        // top of last dimension done, odometer needs resetting
            // which will also reset current to abs_begin
         }
      }
   } while (rollover);
   _used = true;                 // marks as used to allow check for isRolledOver()
   return _current;
}


void SurfaceOdometer::reset()
{
   _begin = _current = _abs_begin;
   _end = _abs_end;
   // collapse least significant dimension first
   _fixedWheel = _current.size()-1;
   _side = false;                // begin at bottom (false)
   _used = false;
   _val = _end[_fixedWheel];     // must store what's being collapsed for retrieval later
   // collapse down to bottom
   _end[_fixedWheel] = _begin[_fixedWheel];
}


int SurfaceOdometer::getSize()
{
   return _size;
}

SurfaceOdometer::~SurfaceOdometer()
{
}
