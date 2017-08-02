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
#include "CheckerboardOdometer.h"
#include "VectorOstream.h"
#include <stdio.h>
#include <stdlib.h>

CheckerboardOdometer::CheckerboardOdometer(Type t, std::vector<int> & begin, std::vector<int> & end, std::vector<int> & sectionSize, std::vector<int> & stepSize)
: _type(t), _begin(begin), _end(end), _sectionSize(sectionSize), _stepSize(stepSize), _isCurrentSectionBegin(true), _used(false)
{
   _current = new std::vector<int>(begin);
   _currentSectionBegin = new std::vector<int>();
   unsigned dims = _begin.size();
   if ((dims == _end.size()) && (dims == _stepSize.size()) && (dims == _sectionSize.size()) ) {}
   else {
      std::cerr<<"Invalid initialization of CheckerboardOdometer: unequal dimensions of initializaiton coordinates:"<<std::endl
         <<"begin dims = "<<dims<<std::endl
         <<"end dims = "<<_end.size()<<std::endl
         <<"step size dims = "<<_stepSize.size()<<std::endl
         <<"section size dims = "<<_sectionSize.size()<<std::endl;
      exit(-1);
   }
   _size = 1;
   for (unsigned i=0; i<dims; ++i) {
      if (t == _PARTIAL_EDGES) _size *= int(float(_end[i]-_begin[i])/float(_stepSize[i]));
      else _size *= int(float(_end[i]-_begin[i]-_sectionSize[i])/float(_stepSize[i]));
      if (_stepSize[i] > _sectionSize[i]) _borders.push_back(_stepSize[i]);
      else _borders.push_back(_sectionSize[i]);
   }
}


bool CheckerboardOdometer::isAtEnd()
{
   return ( (*_current) == _end);
}


bool CheckerboardOdometer::isRolledOver()
{
   return (_used && ( (*_current) == _begin) );
}


std::vector<int> & CheckerboardOdometer::look()
{
   return *_current;
}


std::vector<int> & CheckerboardOdometer::next()
{
   int wheel = _current->size()-1;
   if (_isCurrentSectionBegin) {
      (*_currentSectionBegin) = (*_current);
      std::vector<int>* tmpPtr = _currentSectionBegin;
      _currentSectionBegin = _current;
      _current = tmpPtr;

      for (int i=0;i<=wheel;++i) {
         (*_current)[i]+= (_sectionSize[i]-1);
         if ( (*_current)[i]>_end[i]) (*_current)[i] = _end[i];
      }
      _isCurrentSectionBegin = false;
   }
   else {
      std::vector<int>* tmpPtr = _current;
      _current = _currentSectionBegin;
      _currentSectionBegin = tmpPtr;
      _currentSectionBegin->clear();

      bool keepRolling;
      do {
         (*_current)[wheel]+=_stepSize[wheel];
         keepRolling = false;
         // check if you've rolled over, or if the end of the section will be out of bounds
         if ( ( (*_current)[wheel] > _end[wheel]) ||
         (_type==_NO_PARTIAL_EDGES && (*_current)[wheel]+_sectionSize[wheel] > _end[wheel]) ) {
            (*_current)[wheel] = _begin[wheel];
            if (--wheel >= 0) {  // wheel is < 0 only when incrementing odometer from end to begin
               keepRolling = true;
            }
         }
      } while (keepRolling);
      _isCurrentSectionBegin = true;
   }

   _used = true;
   return *_current;
}


void CheckerboardOdometer::reset()
{
   (*_current) = (*_currentSectionBegin) = _begin;
   _used = false;
}


int CheckerboardOdometer::getSize()
{
   return _size;
}


CheckerboardOdometer::~CheckerboardOdometer()
{
   delete _current;
   delete _currentSectionBegin;
}
