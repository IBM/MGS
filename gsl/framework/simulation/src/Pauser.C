// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005, 2006  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
   std::auto_ptr<TriggerableCaller>& triggerableCaller)
{
   if (functionName != "event") {
      throw SyntaxErrorException(
	 functionName + " is not defined in Pauser as a Triggerable function.");
   } 
   triggerableCaller.reset(
      new PauserEvent(ndpList, &Pauser::event, this));
   return TriggerableBase::_SERIAL;
}
