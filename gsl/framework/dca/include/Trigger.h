// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TRIGGER_H
#define TRIGGER_H
#include "Copyright.h"

#include "Triggerable.h"

#include <deque>
#include <string>
//#include <iostream>
#include <memory>

class TriggerableCaller;
class WorkUnit;

class Trigger 
{
   public:
      virtual bool status()=0;
      virtual void conditionalFire()=0;
      virtual void fireSerial()=0;
      virtual void setDelay(unsigned delay)=0;
      virtual unsigned getDelay() =0;
      virtual void addSerialTriggerableCaller(
	 std::unique_ptr<TriggerableCaller>& triggerableCaller)=0;
      virtual void addParallelTriggerableCaller(
	 std::unique_ptr<TriggerableCaller>& triggerableCaller)=0;
      virtual std::string getDescription()=0;
      virtual void duplicate(std::unique_ptr<Trigger>&& dup) const = 0;
      virtual ~Trigger() {}
      virtual void setNumOfThreads(int numOfThreads) = 0;
      virtual std::deque<WorkUnit*>& getWorkUnits() = 0;
      virtual std::string getPhase() const = 0;
};

#endif
