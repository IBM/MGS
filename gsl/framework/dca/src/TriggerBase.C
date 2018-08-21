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

#include "TriggerBase.h"
#include "Triggerable.h"
#include "TriggerableCaller.h"
#include "TriggerWorkUnit.h"
#include <cassert>
#include <cmath>

TriggerBase::TriggerBase()
   : _delay(0), _state(false), _description("no description"), 
     _numOfThreads(0), _phase("")
{
}

TriggerBase::TriggerBase(const TriggerBase& rv)
   : _delay(rv._delay), _state(rv._state), _stateHistory(rv._stateHistory), 
     _description(rv._description), _numOfThreads(rv._numOfThreads),
     _phase(rv._phase)
{
   copyOwnedHeap(rv);
}

TriggerBase& TriggerBase::operator=(const TriggerBase& rv)
{
   if (&rv != this) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _delay = rv._delay;
      _state = rv._state;
      _stateHistory = rv._stateHistory;
      _description = rv._description;
      _numOfThreads = rv._numOfThreads;
      _phase = rv._phase;
   }
   return *this;
}

void TriggerBase::conditionalFire() 
{
   if (status()) {
      std::vector<TriggerableCaller*>::iterator 
	 it, end = _serialTriggerableCallers.end();
      for (it = _serialTriggerableCallers.begin(); it != end; ++it) {
	 (*it)->event(this);
      }
      end = _parallelTriggerableCallers.end();
      for (it = _parallelTriggerableCallers.begin(); it != end; ++it) {
	 (*it)->event(this);
      }
   }
}

void TriggerBase::fireSerial() 
{
   std::vector<TriggerableCaller*>::iterator 
      it, end = _serialTriggerableCallers.end();
   for (it = _serialTriggerableCallers.begin(); it != end; ++it) {
      (*it)->event(this);
   }
}

TriggerBase::~TriggerBase()
{
   destructOwnedHeap();
}

void TriggerBase::copyOwnedHeap(const TriggerBase& rv)
{
   std::auto_ptr<TriggerableCaller> dup;
   std::vector<TriggerableCaller*>::const_iterator it, 
      end = rv._serialTriggerableCallers.end();
   for (it = rv._serialTriggerableCallers.begin(); it != end; ++it) {
      (*it)->duplicate(dup);
      _serialTriggerableCallers.push_back(dup.release());
   }
   end = rv._parallelTriggerableCallers.end();
   for (it = rv._parallelTriggerableCallers.begin(); it != end; ++it) {
      (*it)->duplicate(dup);
      _parallelTriggerableCallers.push_back(dup.release());
   }
   // workunits should not be copied, instead make your own with the copied
   // TriggerableCallers.
   partitionWorkUnits();
}

void TriggerBase::destructOwnedHeap()
{
   std::vector<TriggerableCaller*>::const_iterator it, 
      end = _serialTriggerableCallers.end();
   for (it = _serialTriggerableCallers.begin(); it != end; ++it) {
      delete (*it);
   }
   _serialTriggerableCallers.clear();
   end = _parallelTriggerableCallers.end();
   for (it = _parallelTriggerableCallers.begin(); it != end; ++it) {
      delete (*it);
   }
   _parallelTriggerableCallers.clear();
   destructWorkUnits();
}

void TriggerBase::destructWorkUnits()
{
   std::deque<WorkUnit*>::iterator it, end = _workUnits.end();
   for(it = _workUnits.begin(); it != end; ++it) {
      delete (*it);
   }
}

void TriggerBase::partitionWorkUnits()
{
   int num = _numOfThreads;
   int totalSize = _parallelTriggerableCallers.size();

   if (num > totalSize) {
      num = totalSize;
   }

   if (num > 0) {
      std::vector<long> partitions;
      partitions.resize(num);
      long chunkSize = long(floor(double(totalSize)/double(num)));

      long extraData = totalSize - (chunkSize * num);
      
      long endIndex = 0;
      for (long i = 0; i < num; ++i) {
	 partitions[i] = endIndex;
	 endIndex = partitions[i] + chunkSize;
	 totalSize -= chunkSize;
	 if (i < extraData) {
	    endIndex += 1;
	    totalSize--;
	 }
      }
      assert(totalSize == 0);
      destructWorkUnits();
      std::vector<TriggerableCaller*>::iterator 
	 begin = _parallelTriggerableCallers.begin();
      for(int i = 0; i < (num-1); ++i) {
 	 _workUnits.push_back(new TriggerWorkUnit(this, begin + partitions[i], 
 						  begin + partitions[i+1]));
      }
      _workUnits.push_back(new TriggerWorkUnit(this, begin + partitions[num-1],
 					       begin + endIndex));
   }
}
