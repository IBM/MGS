// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_typedef_declaration.h"
#include "C_type_specifier.h"
#include "C_declarator.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_typedef_declaration::internalExecute(LensContext *c)
{
   _ts->execute(c);
   _id->execute(c);
}


C_typedef_declaration::C_typedef_declaration(const C_typedef_declaration& rv)
   : C_production(rv), _id(0), _ts(0)
{
   if (rv._ts) {
      _ts = rv._ts->duplicate();
   }
   if (rv._id) {
      _id = rv._id->duplicate();
   }
}


C_typedef_declaration::C_typedef_declaration(
   C_type_specifier *t, C_declarator *d, SyntaxError * error)
   : C_production(error), _id(d), _ts(t)
{
}


C_typedef_declaration* C_typedef_declaration::duplicate() const
{
   return new C_typedef_declaration(*this);
}


C_typedef_declaration::~C_typedef_declaration()
{
   delete _ts;
   delete _id;
}

void C_typedef_declaration::checkChildren() 
{
   if (_id) {
      _id->checkChildren();
      if (_id->isError()) {
         setError();
      }
   }
   if (_ts) {
      _ts->checkChildren();
      if (_ts->isError()) {
         setError();
      }
   }
} 

void C_typedef_declaration::recursivePrint() 
{
   if (_id) {
      _id->recursivePrint();
   }
   if (_ts) {
      _ts->recursivePrint();
   }
   printErrorMessage();
} 
