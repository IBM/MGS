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

// Defines interface for a Stopper
// object, which is used to stop the
#include "Copyright.h"
// simulation
// Author:        Ravi Rao

#ifndef STOPPER_H
#define STOPPER_H

#include "TriggerableBase.h"

#include <list>

class Trigger;
class NDPairList;
class Simulation;

class Stopper : public TriggerableBase
{

   public:
      Stopper(Simulation &s);
      void event(Trigger* trigger, NDPairList* ndPairList);
      ~Stopper() {}

   protected:
      virtual TriggerableBase::EventType createTriggerableCaller(
	 const std::string& functionName, NDPairList* ndpList,
	 std::unique_ptr<TriggerableCaller>&& triggerableCaller);

   private:
      Simulation& _sim;
      std::list<Trigger*> _triggerList;

   public:
      class StopperEvent : public TriggerableCaller
      {
	 public:
	    StopperEvent(
	       NDPairList* ndPairList, 
	       void (Stopper::*triggerFunction)(Trigger*, NDPairList*),
	       Stopper* triggerable) 
	       : TriggerableCaller(ndPairList), 
		 _triggerFunction(triggerFunction),
		 _triggerable(triggerable) {}
	    virtual void event(Trigger* trigger) {
	       (_triggerable->*_triggerFunction)(trigger, _ndPairList);
	    }
	    virtual Triggerable* getTriggerable(){
	       return _triggerable;
	    }
	    virtual void duplicate(std::unique_ptr<TriggerableCaller>&& dup) const {
	       dup.reset(new StopperEvent(*this));
	    }
	    
	 private:
	    void (Stopper::*_triggerFunction)(Trigger*, NDPairList*);
	    Stopper* _triggerable;
      };
};
#endif
