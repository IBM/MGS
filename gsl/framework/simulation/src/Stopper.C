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
