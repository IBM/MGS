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

// Defines interface for a Pauser
// object, which is used to stop the
#include "Copyright.h"
// simulation
// Author:        Ravi Rao

#ifndef PAUSER_H
#define PAUSER_H

#include <list>

#include "TriggerableBase.h"

class Trigger;
class NDPairList;
class Simulation;

class Pauser : public TriggerableBase
{

   public:
      Pauser(Simulation &s);
      void event(Trigger* trigger, NDPairList* ndPairList);
      ~Pauser() {}

   protected:
      virtual TriggerableBase::EventType createTriggerableCaller(
	 const std::string& functionName, NDPairList* ndpList, 
	 std::auto_ptr<TriggerableCaller>& triggerableCaller);

   private:
      Simulation& _sim;

   public:
      class PauserEvent : public TriggerableCaller
      {
	 public:
	    PauserEvent(
	       NDPairList* ndPairList, 
	       void (Pauser::*triggerFunction)(Trigger*, NDPairList*),
	       Pauser* triggerable) 
	       : TriggerableCaller(ndPairList), 
		 _triggerFunction(triggerFunction),
		 _triggerable(triggerable) {}
	    virtual void event(Trigger* trigger) {
	       (_triggerable->*_triggerFunction)(trigger, _ndPairList);
	    }
	    virtual Triggerable* getTriggerable(){
	       return _triggerable;
	    }
	    virtual void duplicate(std::auto_ptr<TriggerableCaller>& dup) const {
	       dup.reset(new PauserEvent(*this));
	    }
	    
	 private:
	    void (Pauser::*_triggerFunction)(Trigger*, NDPairList*);
	    Pauser* _triggerable;
      };
};
#endif
