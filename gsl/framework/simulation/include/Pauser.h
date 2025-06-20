// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
	 std::unique_ptr<TriggerableCaller>&& triggerableCaller);

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
	    virtual void duplicate(std::unique_ptr<TriggerableCaller>&& dup) const {
	       dup.reset(new PauserEvent(*this));
	    }
	    
	 private:
	    void (Pauser::*_triggerFunction)(Trigger*, NDPairList*);
	    Pauser* _triggerable;
      };
};
#endif
