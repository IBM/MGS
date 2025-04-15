// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "VectoredCheckerboardOdometer.h"
#include "CheckerboardOdometer.h"
#include "VectorOstream.h"
#include <stdio.h>
#include <stdlib.h>

VectoredCheckerboardOdometer::VectoredCheckerboardOdometer(CheckerboardOdometer::Type t, std::vector<int> & begin,
std::vector<int> & end, std::vector<int> & sectionSize,
std::vector<int> & stepSize,std::vector<int> & startPos)
: CheckerboardOdometer(t, begin, end, sectionSize, stepSize), _startPos(startPos)
{
   _current->clear();            // will initialize difference for Vectored variety below
   unsigned dims = _begin.size();
   if (dims == _startPos.size()) {}
   else {
      std::cerr<<"Invalid initialization of VectoredCheckerboardOdometer: unequal dimensions of initializaiton coordinates:"<<std::endl
         <<"start pos dims = "<<_startPos.size()<<std::endl;
      exit(-1);
   }

   for (unsigned i=0; i<dims; ++i) {
      if (_startPos[i] + _sectionSize[i] - 1 < _begin[i] || _startPos[i] > _end[i]) {
         std::cerr<<"Start position is out of bounds on VectoredCheckerboardOdometer!"<<std::endl;
         exit(-1);
      }
      if (_type == _NO_PARTIAL_EDGES) {
         if (_startPos[i] < _begin[i]) _current->push_back(_begin[i]);
         else if (_startPos[i] + _sectionSize[i] - 1 > _end[i]) _current->push_back(_end[i] - _sectionSize[i] + 1);
         else _current->push_back(_startPos[i]);
         _offsets.push_back(0);  // initialize offset vector
      }
      else if (_startPos[i] >= _begin[i]) {
         _current->push_back(_startPos[i]);
         _offsets.push_back(0);
      }
      else {
         _current->push_back(_begin[i]);
         _offsets.push_back(_startPos[i] - _begin[i]);
      }
   }
}


bool VectoredCheckerboardOdometer::isAtEnd()
{
   return ( (*_current) == _end);
}


bool VectoredCheckerboardOdometer::isRolledOver()
{
   return (_used && ( (*_current) == _begin) );
}


std::vector<int> & VectoredCheckerboardOdometer::look()
{
   return *_current;
}


std::vector<int> & VectoredCheckerboardOdometer::next()
{
   int dims;
   int wheel = dims = _current->size()-1;
   if (_isCurrentSectionBegin) {
      (*_currentSectionBegin) = (*_current);
      std::vector<int>* tmpPtr = _currentSectionBegin;
      _currentSectionBegin = _current;
      _current = tmpPtr;

      for (int i=0;i<=wheel;++i) {
         int& currentWheelValue = (*_current)[i];
         currentWheelValue += ( _sectionSize[i] - 1 + _offsets[i] );
         if ( currentWheelValue>_end[i]) currentWheelValue = _end[i];
      }
      _isCurrentSectionBegin = false;
   }
   else {
      std::vector<int>* tmpPtr = _current;
      _current = _currentSectionBegin;
      _currentSectionBegin = tmpPtr;
      _currentSectionBegin->clear();

      bool keepRolling;
      while (wheel>=0) {
         do {
            int & currentWheelValue = (*_current)[wheel];
            int & offset = _offsets[wheel];
            int stepSize = _stepSize[dims-wheel];
            int begin = _begin[wheel];
            int end = _end[wheel];
            int sectionSize = _sectionSize[wheel];
            int nextEnd = currentWheelValue+sectionSize-1;

            currentWheelValue+= stepSize;
            keepRolling = false;

            /* * check if you've rolled over positively * */
            if ( ( currentWheelValue > end) || (_type==_NO_PARTIAL_EDGES && nextEnd > end) ) {
               if (_type == _PARTIAL_EDGES) {
                  if (currentWheelValue - stepSize + sectionSize - 1 > end) {
                     offset = end - sectionSize + 1 + stepSize - currentWheelValue;
                     //offset = currentWheelValue - end - stepSize - 1;
                     currentWheelValue = begin;
                  }
                  else {
                     offset = 0;
                     // currentWheelValue = begin + currentWheelValue - end;
                     currentWheelValue = begin + end - currentWheelValue + stepSize;
                  }
               }
               else currentWheelValue = begin;
               if (--wheel >= 0) keepRolling = true;
            }

            /* * check if you've rolled over negatively * */
            else if ( ( nextEnd < begin) || (_type==_NO_PARTIAL_EDGES && currentWheelValue < begin) ) {
               if (_type == _PARTIAL_EDGES) {
                  offset = 0;
                  // currentWheelValue = end - sectionSize + 1 - begin + currentWheelValue;
                  currentWheelValue = end - sectionSize + 1 + begin + currentWheelValue - stepSize;
               }
               else {
                  currentWheelValue = end - sectionSize + 1;
                  offset = 0;
               }
               if (--wheel >= 0) keepRolling = true;
            }

            /* * check if only offset needs modifying * */
            else if ( stepSize > 0 && offset != 0 ) {
               offset += stepSize;
               if (offset <= 0) currentWheelValue = begin;
               else {
                  currentWheelValue = offset;
                  offset = 0;
               }
            }

            else if ( currentWheelValue >= begin && offset != 0) {
               offset -= stepSize;
               if (offset > 0) offset = 0;
            }

            /* * check if only begin is out of bounds * */
            else if ( currentWheelValue < begin) {
                                 // first time out of bounds this pass
               if (offset == 0) {
                  offset =  currentWheelValue - begin;
                  currentWheelValue = begin;
               }
               else {            // not the first time out of bounds this pass
                  offset += stepSize;
                  if (offset <= -sectionSize) {
                     offset -= stepSize;
                     currentWheelValue = end - sectionSize - offset + 1;
                     if (--wheel >= 0) keepRolling = true;
                  }
                  else currentWheelValue = begin;
               }
            }

         } while (keepRolling);
         --wheel;
      }
      _isCurrentSectionBegin = true;
   }

   _used = true;
   return *_current;
}

void VectoredCheckerboardOdometer::reset()
{
   (*_current) = (*_currentSectionBegin) = _startPos;
   _used = false;
}

int VectoredCheckerboardOdometer::getSize()
{
   std::cerr<<"Size not computed on VectoredCheckerboardOdometer!"<<std::endl;
   return 0;
}

VectoredCheckerboardOdometer::~VectoredCheckerboardOdometer()
{
}
