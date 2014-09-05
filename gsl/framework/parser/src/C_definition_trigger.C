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

#include "C_definition_trigger.h"
#include "LensContext.h"
#include "C_parameter_type.h"
#include "C_parameter_type_list.h"
#include "C_declarator.h"
#include "C_trigger.h"
#include "Simulation.h"
#include "TriggerType.h"
#include "TriggerTypeDataItem.h"
#include "CompositeTriggerDescriptor.h"
#include "TypeRegistry.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

#include <memory>

void C_definition_trigger::internalExecute(LensContext *c)
{
   // call executes
   _declarator->execute(c);
   _constructor_ptl->execute(c);

   TriggerType* tt = c->sim->getTriggerType(_declarator->getName());
   _instanceFactory = dynamic_cast<InstanceFactory*>(tt);
   if (_instanceFactory == 0) {
      std::string mes = 
	 "dynamic cast of TriggerType to InstanceFactory failed";
      throwError(mes);
   }
   _constructor_list = _constructor_ptl->getList();
   TriggerTypeDataItem* ttdi = new TriggerTypeDataItem;
   ttdi->setTriggerType(tt);
   std::auto_ptr<DataItem> diap(ttdi);
   try {
      c->symTable.addEntry(_declarator->getName(), diap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining trigger, " + e.getError());
   }
}

C_definition_trigger::C_definition_trigger(const C_definition_trigger& rv)
   : C_definition(rv), _declarator(0), _constructor_ptl(0), 
     _instanceFactory(rv._instanceFactory), _constructor_list(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._constructor_ptl) {
      _constructor_ptl = rv._constructor_ptl->duplicate();
   }
   if (rv._constructor_list) {
      _constructor_list = rv._constructor_list;
   }
}

C_definition_trigger::C_definition_trigger(
   C_declarator *d, C_parameter_type_list *p, SyntaxError * error)
   : C_definition(error), _declarator(d), _constructor_ptl(p), 
     _instanceFactory(0), _constructor_list(0)
{
}

C_definition_trigger* C_definition_trigger::duplicate() const
{
   return new C_definition_trigger(*this);
}

std::string C_definition_trigger::getDeclarator()
{
   return _declarator->getName();
}

std::list<C_parameter_type>* C_definition_trigger::getConstructorParams()
{
   return _constructor_list;
}

InstanceFactory *C_definition_trigger::getInstanceFactory()
{
   return _instanceFactory;
}

C_definition_trigger::~C_definition_trigger()
{
   delete _constructor_ptl;
   delete _declarator;
}

void C_definition_trigger::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_constructor_ptl) {
      _constructor_ptl->checkChildren();
      if (_constructor_ptl->isError()) {
         setError();
      }
   }
} 

void C_definition_trigger::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_constructor_ptl) {
      _constructor_ptl->recursivePrint();
   }
   printErrorMessage();
} 
