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

#include "C_declaration_trigger.h"
#include "Trigger.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_trigger.h"
#include "TriggerDataItem.h"
#include "Simulation.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include <cassert>

void C_declaration_trigger::internalExecute(LensContext *c)
{
   // For now, remove later (don't forget the header cassert)
   assert(_trigger != 0);

   _declarator->execute(c);
   _trigger->execute(c);

   TriggerDataItem *nsdi = new TriggerDataItem;
   std::unique_ptr<DataItem> nsdi_ap(nsdi);

   Trigger* t;
   t = _trigger->getTrigger();

   nsdi->setTrigger(t);

   try {
      c->symTable.addEntry(_declarator->getName(), nsdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring trigger, " + e.getError());
   }
}


C_declaration_trigger* C_declaration_trigger::duplicate() const
{
   return new C_declaration_trigger(*this);
}


C_declaration_trigger::C_declaration_trigger(
   C_declarator *d, C_trigger *t, SyntaxError * error)
   : C_declaration(error), _declarator(d), _trigger(t), _name(0)
{
   _name = new std::string("");
}

C_declaration_trigger::C_declaration_trigger(const C_declaration_trigger& rv)
   : C_declaration(rv), _declarator(0), _trigger(0), _name(0)
{
   if (rv._name) {
      _name = new std::string(*(rv._name));
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._trigger) {
      _trigger = rv._trigger->duplicate();
   }
}


C_declaration_trigger::~C_declaration_trigger()
{
   delete _declarator;
   delete _trigger;
   delete _name;
}

void C_declaration_trigger::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_trigger) {
      _trigger->checkChildren();
      if (_trigger->isError()) {
         setError();
      }
   }
} 

void C_declaration_trigger::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_trigger) {
      _trigger->recursivePrint();
   }
   printErrorMessage();
} 
