// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
	 std::auto_ptr<TriggerableCaller>& triggerableCaller)=0;
      virtual void addParallelTriggerableCaller(
	 std::auto_ptr<TriggerableCaller>& triggerableCaller)=0;
      virtual std::string getDescription()=0;
      virtual void duplicate(std::auto_ptr<Trigger>& dup) const = 0;
      virtual ~Trigger() {}
      virtual void setNumOfThreads(int numOfThreads) = 0;
      virtual std::deque<WorkUnit*>& getWorkUnits() = 0;
      virtual std::string getPhase() const = 0;
};

#endif
