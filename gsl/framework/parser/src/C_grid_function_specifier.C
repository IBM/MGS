// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_grid_function_specifier.h"
#include "C_directive.h"
#include "C_grid_function_name.h"
#include "SyntaxError.h"
#include "C_production_grid.h"

void C_grid_function_specifier::internalExecute(GslContext *c, Grid* g)
{
   if (_directive) _directive->execute(c);
   else {
     _gridFuncName->execute(c, g);
   }
}


C_grid_function_specifier::C_grid_function_specifier(
   const C_grid_function_specifier& rv)
   : C_production_grid(rv), _directive(0), _gridFuncName(0)
{
   if (rv._directive) {
      _directive = rv._directive->duplicate();
   }
   else if (rv._gridFuncName) {
      _gridFuncName = rv._gridFuncName->duplicate();
   }
}


C_grid_function_specifier::C_grid_function_specifier(
   C_directive *f, SyntaxError * error)
   : C_production_grid(error), _directive(f), _gridFuncName(0)
{
}


C_grid_function_specifier::C_grid_function_specifier(C_grid_function_name *g, 
						     SyntaxError * error)
   : C_production_grid(error), _directive(0), _gridFuncName(g)
{
}


C_grid_function_specifier* C_grid_function_specifier::duplicate() const
{
   return new C_grid_function_specifier(*this);
}


C_grid_function_specifier::~C_grid_function_specifier()
{
   delete _directive;
   delete _gridFuncName;
}

void C_grid_function_specifier::checkChildren() 
{
   if (_directive) {
      _directive->checkChildren();
      if (_directive->isError()) {
         setError();
      }
   }
   if (_gridFuncName) {
      _gridFuncName->checkChildren();
      if (_gridFuncName->isError()) {
         setError();
      }
   }
} 

void C_grid_function_specifier::recursivePrint() 
{
   if (_directive) {
      _directive->recursivePrint();
   }
   if (_gridFuncName) {
      _gridFuncName->recursivePrint();
   }
   printErrorMessage();
} 
