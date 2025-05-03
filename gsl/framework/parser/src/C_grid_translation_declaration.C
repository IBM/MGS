// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_grid_translation_declaration.h"
#include "C_declaration.h"
#include "C_grid_function_specifier.h"
#include "SyntaxError.h"
#include "C_production_grid.h"

void C_grid_translation_declaration::internalExecute(GslContext *c, Grid* g)
{
   if (_declaration) _declaration->execute(c);
   else _gridFuncSpec->execute(c, g);
}


C_grid_translation_declaration::C_grid_translation_declaration(
   const C_grid_translation_declaration& rv)
   : C_production_grid(rv), _declaration(0), _gridFuncSpec(0)
{
   if (rv._declaration) {
      _declaration = rv._declaration->duplicate();
   }
   if (rv._gridFuncSpec) {
      _gridFuncSpec = rv._gridFuncSpec->duplicate();
   }
}


C_grid_translation_declaration::C_grid_translation_declaration(
   C_declaration *d, SyntaxError * error)
   : C_production_grid(error), _declaration(d), _gridFuncSpec(0)
{
}


C_grid_translation_declaration::C_grid_translation_declaration(
   C_grid_function_specifier *g, SyntaxError * error)
   : C_production_grid(error), _declaration(0), _gridFuncSpec(g)
{
}


C_grid_translation_declaration* 
C_grid_translation_declaration::duplicate() const
{
   return new C_grid_translation_declaration(*this);
}


C_grid_translation_declaration::~C_grid_translation_declaration()
{
   delete _declaration;
   delete _gridFuncSpec;
}

void C_grid_translation_declaration::checkChildren() 
{
   if (_declaration) {
      _declaration->checkChildren();
      if (_declaration->isError()) {
         setError();
      }
   }
   if (_gridFuncSpec) {
      _gridFuncSpec->checkChildren();
      if (_gridFuncSpec->isError()) {
         setError();
      }
   }
} 

void C_grid_translation_declaration::recursivePrint() 
{
   if (_declaration) {
      _declaration->recursivePrint();
   }
   if (_gridFuncSpec) {
      _gridFuncSpec->recursivePrint();
   }
   printErrorMessage();
} 
