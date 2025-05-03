// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_grid_translation_unit.h"
#include "C_grid_translation_declaration_list.h"
#include "SyntaxError.h"
#include "C_production_grid.h"

void C_grid_translation_unit::internalExecute(GslContext *c, Grid* g)
{
   _gl->execute(c, g);
}


C_grid_translation_unit::C_grid_translation_unit(
   const C_grid_translation_unit& rv)
   :  C_production_grid(rv), _gl(0), _tdError(0)
{
   if (rv._gl) {
      _gl = rv._gl->duplicate();
   }
   if (rv._tdError) {
      _tdError = rv._tdError->duplicate();
   }
}


C_grid_translation_unit::C_grid_translation_unit(
   C_grid_translation_declaration_list *g, SyntaxError * error)
   :  C_production_grid(error), _gl(g), _tdError(0)
{
}


C_grid_translation_unit* C_grid_translation_unit::duplicate() const
{
   return new C_grid_translation_unit(*this);
}


C_grid_translation_unit::~C_grid_translation_unit()
{
   delete _gl;
   delete _tdError;
}

void C_grid_translation_unit::checkChildren() 
{
   if (_gl) {
      _gl->checkChildren();
      if (_gl->isError()) {
         setError();
      }
   }
} 

void C_grid_translation_unit::recursivePrint() 
{
   if (_gl) {
      _gl->recursivePrint();
   }
   printErrorMessage();
} 

void C_grid_translation_unit::printTdError() 
{
   _tdError->setOriginal();
   _tdError->setError(true);   
   if (_tdError) {
      _tdError->printMessage();
   }
}
