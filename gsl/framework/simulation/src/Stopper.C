// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Stopper.h"
#include "Simulation.h"
#include "SyntaxErrorException.h"

Stopper::Stopper(Simulation& s): _sim(s)
{
}

void Stopper::event(Trigger* trigger, NDPairList* ndPairList)
{
   if (_sim.getRank()==0) printf("Stopping Simulation.\n");
   _sim.detachUserInterface();
   _sim.stop();
}

TriggerableBase::EventType Stopper::createTriggerableCaller(
   const std::string& functionName, NDPairList* ndpList,
   std::unique_ptr<TriggerableCaller>&& triggerableCaller)
{
   if (functionName != "event") {
      throw SyntaxErrorException(
	 functionName + " is not defined in Stopper as a Triggerable function.");
   } 
   triggerableCaller.reset(
      new StopperEvent(ndpList, &Stopper::event, this));
   return TriggerableBase::_SERIAL;
}
