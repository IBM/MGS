// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_trigger.h"
#include "C_query_path_product.h"
#include "C_declarator.h"
#include "TriggerType.h"
#include "TriggerDataItem.h"
#include "Trigger.h"
#include "Publisher.h"
#include "LensContext.h"
#include "CompositeTrigger.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

#include <assert.h>

void C_trigger::internalExecute(LensContext *c)
{
   if (_queryPathProduct) _queryPathProduct->execute(c);
   if (_declarator) _declarator->execute(c);
   if (_ct1) _ct1->execute(c);
   if (_ct2) _ct2->execute(c);

   if (_queryPathProduct) {
      TriggerType* tt = _queryPathProduct->getTriggerDescriptor();
      std::vector<DataItem*> v;
      _trigger = tt->getTrigger(v);
      return;
   }
   else if (_declarator) {
      const DataItem* di = c->symTable.getEntry(_declarator->getName());
      const TriggerDataItem* tdi = dynamic_cast<const TriggerDataItem*>(di);
      if (tdi == 0) {
	 std::string mes = 
	    "dynamic cast of DataItem to TriggerDataItem failed";
	 throwError(mes);
      }
      _trigger = tdi->getTrigger();
      return;
   }
   std::string op;
   switch(_type) {
   case _SINGLE:
      if (!_ct1) {
	 std::string mes = "Internal Error: _ct1 is 0 for _SINGLE";
	 throwError(mes);
      }
      _trigger = _ct1->getTrigger();
      return;
      break;
   case _AND:
      op = "AND";
      break;
   case _OR:
      op = "OR";
      break;
   case _XOR:
      op = "XOR";
      break;
   default:
      std::string mes = "Internal Error: default case run";
      throwError(mes);
   }
   if (!_ct1) {
      std::string mes = "Internal Error: _ct1 is 0";
      throwError(mes);
   }
   if (!_ct2) {
      std::string mes = "Internal Error: _ct2 is 0";
      throwError(mes);
   }
   
   _trigger = new CompositeTrigger(*(c->sim), _ct1->getTrigger(), 
				   _ct2->getTrigger(), op);
   // Assing ownership to symTable
   TriggerDataItem *nsdi = new TriggerDataItem;
   std::unique_ptr<DataItem> nsdi_ap(nsdi);
   nsdi->setTrigger(_trigger);
   c->symTable.addCompositeTrigger(nsdi_ap);
}


C_trigger::C_trigger(C_query_path_product* qpp, SyntaxError * error)
   : C_production(error), _queryPathProduct(qpp), _declarator(0), _trigger(0), 
     _ct1(0), _ct2(0), _type(_BASIC)
{
}


C_trigger::C_trigger(C_declarator* n, SyntaxError * error)
   : C_production(error), _queryPathProduct(0), _declarator(n), _trigger(0), 
     _ct1(0), _ct2(0), _type(_BASIC)
{
}

C_trigger::C_trigger(const C_trigger& rv)
   : C_production(rv), _queryPathProduct(0), _declarator(0), 
     _trigger(rv._trigger), _ct1(0), _ct2(0), _type(rv._type)
{
   if (rv._queryPathProduct) {
      _queryPathProduct = rv._queryPathProduct->duplicate();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._ct1) {
      _ct1 = rv._ct1->duplicate();
   }
   if (rv._ct2) {
      _ct2 = rv._ct2->duplicate();
   }   
}

C_trigger::C_trigger(C_trigger *t, Type type, SyntaxError * error)
   : C_production(error), _queryPathProduct(0), _declarator(0), _trigger(0), 
     _ct1(t), _ct2(0), _type(type)
{
}

C_trigger::C_trigger(C_trigger *t1, C_trigger *t2, Type type, 
		     SyntaxError * error)
   : C_production(error), _queryPathProduct(0), _declarator(0), _trigger(0), 
     _ct1(t1), _ct2(t2), _type(type)
{
}

C_trigger* C_trigger::duplicate() const
{
   return new C_trigger(*this);
}

C_trigger::~C_trigger()
{
   delete _queryPathProduct;
   delete _declarator;
   delete _ct1;
   delete _ct2;
}

void C_trigger::checkChildren() 
{
   if (_queryPathProduct) {
      _queryPathProduct->checkChildren();
      if (_queryPathProduct->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_ct1) {
      _ct1->checkChildren();
      if (_ct1->isError()) {
         setError();
      }
   }
   if (_ct2) {
      _ct2->checkChildren();
      if (_ct2->isError()) {
         setError();
      }
   }
} 

void C_trigger::recursivePrint() 
{
   if (_queryPathProduct) {
      _queryPathProduct->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_ct1) {
      _ct1->recursivePrint();
   }
   if (_ct2) {
      _ct2->recursivePrint();
   }
   printErrorMessage();
} 
