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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "VectoredOdometer.h"
#include "VectorOstream.h"
#include <stdio.h>
#include <stdlib.h>

VectoredOdometer::VectoredOdometer(std::vector<int> & begin, std::vector<int> & end, std::vector<int> & steps)
: _begin(begin),_current(begin), _end(end), _steps(steps), _atEdge(false), _rolledOver(false)
{
   if (_begin.size() == 0) _size = 0;
   else {
      _size = 1;
      unsigned end = _begin.size();
      for (unsigned i = 0; i<end; ++i) {
         int dim = _end[i]-_begin[i];
         _size *= (dim+1);
         _dims.push_back(dim);
         if (_current[i]+_steps[i]>_end[i]) _atEdge = true;
      }
   }
}

void VectoredOdometer::set(std::vector<int> & newCurrent)
{
   _current = newCurrent;
   _atEdge = false;
   unsigned end = _begin.size();
   for (unsigned i = 0; i<end; ++i)
      if (_current[i]+_steps[i]>_end[i]) _atEdge = true;
   _rolledOver = false;
}

bool VectoredOdometer::isAtEnd()
{
   return _atEdge;
}

bool VectoredOdometer::isRolledOver()
{
   return _rolledOver;
}

std::vector<int> & VectoredOdometer::look()
{
   return _current;
}

std::vector<int> & VectoredOdometer::next()
{
   _rolledOver=false;
   unsigned endWheel = _current.size()-1;
   for (unsigned wheel = 0;wheel!=endWheel;++wheel) {
      int& current = _current[wheel];
      current += _steps[wheel];
      if (current > _end[wheel]) {
         _atEdge=false;
         _rolledOver=true;
         current -= _dims[wheel];
      }
      else if (current+_steps[wheel]>_end[wheel]) _atEdge=true;
   }
   return _current;
}

void VectoredOdometer::reset()
{
   _current = _begin;
   _atEdge = false;
   unsigned end = _begin.size();
   for (unsigned i = 0; i<end; ++i)
      if (_current[i]+_steps[i]>_end[i]) _atEdge = true;
   _rolledOver = false;
}

int VectoredOdometer::getSize()
{
   return _size;
}

VectoredOdometer::~VectoredOdometer()
{
}
