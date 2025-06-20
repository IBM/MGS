// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Pauser.h"
#include "Simulation.h"
#include "SyntaxErrorException.h"

Pauser::Pauser(Simulation& s): _sim(s)
{
}

void Pauser::event(Trigger* trigger, NDPairList* ndPairList)
{
   if (_sim.getPauserStatus()) {
      if (_sim.getRank()==0) std::cout<<"Simulation paused at iteration number "<<_sim.getIteration()<<"." << std::endl ;
      _sim.pause();
   }
}

TriggerableBase::EventType Pauser::createTriggerableCaller(
   const std::string& functionName, NDPairList* ndpList,
   std::unique_ptr<TriggerableCaller>&& triggerableCaller)
{
   if (functionName != "event") {
      throw SyntaxErrorException(
	 functionName + " is not defined in Pauser as a Triggerable function.");
   } 
   triggerableCaller.reset(
      new PauserEvent(ndpList, &Pauser::event, this));
   return TriggerableBase::_SERIAL;
}
