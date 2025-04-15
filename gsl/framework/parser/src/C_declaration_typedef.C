// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_typedef.h"
#include "C_typedef_declaration.h"
#include "LensContext.h"
#include "SyntaxError.h"

void C_declaration_typedef::internalExecute(LensContext *c)
{
   _t->execute(c);
}


C_declaration_typedef* C_declaration_typedef::duplicate() const
{
   return new C_declaration_typedef(*this);
}


C_declaration_typedef::C_declaration_typedef(C_typedef_declaration *d, SyntaxError * error)
   : C_declaration(error), _t(d)
{
}


C_declaration_typedef::C_declaration_typedef(const C_declaration_typedef& rv)
   : C_declaration(rv), _t(0)
{
   if (rv._t) {
      _t = rv._t->duplicate();
   }
}


C_declaration_typedef::~C_declaration_typedef()
{
   delete _t;
}

void C_declaration_typedef::checkChildren() 
{
   if (_t) {
      _t->checkChildren();
      if (_t->isError()) {
         setError();
      }
   }
} 

void C_declaration_typedef::recursivePrint() 
{
   if (_t) {
      _t->recursivePrint();
   }
   printErrorMessage();
} 
