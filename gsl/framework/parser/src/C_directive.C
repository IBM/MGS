// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_directive.h"
#include "C_functor_specifier.h"
#include "C_trigger_specifier.h"
#include "C_system_call.h"
#include "SyntaxError.h"
#include "C_production.h"

const DataItem* C_directive::getRVal() const
{
   const DataItem* retval = 0;
   if (_functorSpec) {
      retval = _functorSpec->getRVal();
   }
   return retval;
}

void C_directive::internalExecute(GslContext *c)
{
   if (_functorSpec) {
      _functorSpec->execute(c);
   }
   if (_triggerSpec) {
      _triggerSpec->execute(c);
   }
   if (_systemCall) {
      _systemCall->execute(c);
   }
}

C_directive::C_directive(const C_directive& rv)
   : C_production(rv), _functorSpec(0), _triggerSpec(0), _systemCall(0)
{
   if (rv._functorSpec) {
      _functorSpec = rv._functorSpec->duplicate();
   }
   if (rv._triggerSpec) {
      _triggerSpec = rv._triggerSpec->duplicate();
   }
   if (rv._systemCall) {
      _systemCall = rv._systemCall->duplicate();
   }
}

C_directive::C_directive(C_functor_specifier *fs, SyntaxError * error)
   : C_production(error), _functorSpec(fs), _triggerSpec(0), _systemCall(0)
{
}

C_directive::C_directive(C_trigger_specifier *ts, SyntaxError * error)
   : C_production(error), _functorSpec(0), _triggerSpec(ts), _systemCall(0)
{
}

C_directive::C_directive(C_system_call *sc, SyntaxError * error)
   : C_production(error), _functorSpec(0), _triggerSpec(0), _systemCall(sc)
{
}

C_directive* C_directive::duplicate() const
{
   return new C_directive(*this);
}

C_directive::~C_directive()
{
   delete _functorSpec;
   delete _triggerSpec;
   delete _systemCall;
}

void C_directive::checkChildren() 
{
   if (_functorSpec) {
      _functorSpec->checkChildren();
      if (_functorSpec->isError()) {
         setError();
      }
   }
   if (_triggerSpec) {
      _triggerSpec->checkChildren();
      if (_triggerSpec->isError()) {
         setError();
      }
   }
   if (_systemCall) {
      _systemCall->checkChildren();
      if (_systemCall->isError()) {
         setError();
      }
   }
} 

void C_directive::recursivePrint() 
{
   if (_functorSpec) {
      _functorSpec->recursivePrint();
   }
   if (_triggerSpec) {
      _triggerSpec->recursivePrint();
   }
   if (_systemCall) {
      _systemCall->recursivePrint();
   }
   printErrorMessage();
} 
