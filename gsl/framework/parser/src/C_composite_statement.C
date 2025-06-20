// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_composite_statement.h"
#include "C_declaration.h"
#include "C_directive.h"
#include <typeinfo>
#include "GslContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"


void C_composite_statement::internalExecute(GslContext *c)
{
   if(_declaration)
      _declaration->execute(c);
   else if(_directive)
      _directive->execute(c);
   else {
      std::string mes = "Composite statement is empty!";
      throwError(mes);
   }
}


C_composite_statement::C_composite_statement(const C_composite_statement& rv)
   : C_production(rv), _declaration(0), _directive(0)
{
   if (rv._declaration) {
      _declaration = rv._declaration->duplicate();
   }
   if (rv._directive) {
      _directive = rv._directive->duplicate();
   }
}


C_composite_statement::C_composite_statement(
   C_declaration *d, SyntaxError * error)
   : C_production(error), _declaration(d), _directive(0)
{
}


C_composite_statement::C_composite_statement(
   C_directive *f, SyntaxError * error)
   : C_production(error), _declaration(0), _directive(f)
{
}


C_composite_statement* C_composite_statement::duplicate() const
{
   return new C_composite_statement(*this);
}


C_composite_statement::~C_composite_statement()
{
   delete _declaration;
   delete _directive;
}

void C_composite_statement::checkChildren() 
{
   if (_declaration) {
      _declaration->checkChildren();
      if (_declaration->isError()) {
         setError();
      }
   }
   if (_directive) {
      _directive->checkChildren();
      if (_directive->isError()) {
         setError();
      }
   }
} 

void C_composite_statement::recursivePrint() 
{
   if (_declaration) {
      _declaration->recursivePrint();
   }
   if (_directive) {
      _directive->recursivePrint();
   }
   printErrorMessage();
} 
