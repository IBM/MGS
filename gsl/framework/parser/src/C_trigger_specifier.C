// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_trigger_specifier.h"
#include "C_declarator.h"
#include "C_trigger.h"
#include "Trigger.h"
#include "Triggerable.h"
#include "TriggerDataItem.h"
#include "TriggerableDataItem.h"
#include "Pauser.h"
#include "Stopper.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"
#include "C_ndpair_clause_list.h"

#include "NDPairList.h"
#include "Simulation.h"
#include <vector>

void C_trigger_specifier::internalExecute(LensContext *c)
{

   if (_trigger) {
      _trigger->execute(c);
   } else {
      std::string mes = "Internal Error: _trigger is 0 in C_trigger_specifier::internalExecute";
      throwError(mes);
   }

   if (_triggerable) {
      _triggerable->execute(c);

      if (_action) {
	 _action->execute(c);
      } else {
	 std::string mes = "Internal Error: _action is 0 in C_trigger_specifier::internalExecute";
	 throwError(mes);	 
      }
      NDPairList* ndpList = 0;
      std::auto_ptr<NDPairList> ndpListAp;
      ndpListAp.reset(0);
      if (_ndpairList) {
	 _ndpairList->execute(c);
	 ndpList = _ndpairList->getList();
      } 

      DataItem* di = c->symTable.getEntry(_triggerable->getName());
      TriggerableDataItem* tdi = dynamic_cast<TriggerableDataItem*>(di);
      if (tdi == 0) {
	 std::string mes = "dynamic cast of DataItem to TriggerableDataItem failed";
	 throwError(mes);
      }

      Trigger* trigger = _trigger->getTrigger();

      std::vector<Triggerable*> triggerables;
      triggerables = tdi->getTriggerables();
      std::vector<Triggerable*>::iterator it, end = triggerables.end();
      for(it = triggerables.begin(); it != end; ++it) {
	 if (ndpList) {
	    ndpList->duplicate(ndpListAp);
	 }
	 (*it)->addTrigger(trigger, _action->getName(), ndpListAp);
      }
   } else {
      Triggerable* triggerable = 0;
      std::auto_ptr<NDPairList> ndpList;
      ndpList.reset(0);
      if (_triggerableSpecifier == "pause") {
	 triggerable = c->sim->getPauser();
	 c->sim->setPauserStatus(true);
      }
      if (_triggerableSpecifier == "stop") {
	 triggerable = c->sim->getStopper();
      }
      Trigger* trigger = _trigger->getTrigger();
      triggerable->addTrigger(trigger, "event", ndpList);
   }
}


C_trigger_specifier::C_trigger_specifier(
   C_declarator* triggerable, C_declarator* action, 
   C_ndpair_clause_list* ndpairList, C_trigger* trigger, SyntaxError * error)
   : C_production(error), _triggerable(triggerable), _action(action), 
     _trigger(trigger), _ndpairList(ndpairList)
{
}


C_trigger_specifier::C_trigger_specifier(
   std::string triggerableSpecifier, C_trigger* trigger,
   SyntaxError * error)
   : C_production(error), _triggerable(0), _action(0), _trigger(trigger), 
     _ndpairList(0), _triggerableSpecifier(triggerableSpecifier)
{
}


C_trigger_specifier::C_trigger_specifier(const C_trigger_specifier& rv)
   : C_production(rv), _triggerable(0), _action(0), _trigger(0), 
     _ndpairList(0), _triggerableSpecifier(rv._triggerableSpecifier)
{
   if (rv._trigger) {
      _trigger = rv._trigger->duplicate();
   }
   if (rv._action) {
      _action = rv._action->duplicate();
   }
   if (rv._triggerable) {
      _triggerable = rv._triggerable->duplicate();
   }
   if (rv._ndpairList) {
      _ndpairList = rv._ndpairList->duplicate();
   }
}


C_trigger_specifier* C_trigger_specifier::duplicate() const
{
   return new C_trigger_specifier(*this);
}


C_trigger_specifier::~C_trigger_specifier()
{
   delete _triggerable;
   delete _action;
   delete _trigger;
   delete _ndpairList;
}

void C_trigger_specifier::checkChildren() 
{
   if (_triggerable) {
      _triggerable->checkChildren();
      if (_triggerable->isError()) {
         setError();
      }
   }
   if (_action) {
      _action->checkChildren();
      if (_action->isError()) {
         setError();
      }
   }
   if (_trigger) {
      _trigger->checkChildren();
      if (_trigger->isError()) {
         setError();
      }
   }
   if (_ndpairList) {
      _ndpairList->checkChildren();
      if (_ndpairList->isError()) {
         setError();
      }
   }
} 

void C_trigger_specifier::recursivePrint() 
{
   if (_triggerable) {
      _triggerable->recursivePrint();
   }
   if (_action) {
      _action->recursivePrint();
   }
   if (_trigger) {
      _trigger->recursivePrint();
   }
   if (_ndpairList) {
      _ndpairList->recursivePrint();
   }
   printErrorMessage();
} 
